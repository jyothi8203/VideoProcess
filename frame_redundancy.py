import glob
import cv2 as cv
import os
import numpy as np
from datetime import datetime
import csv,json,logger,logging
import numba as nb
from numba import njit, prange
from process_text import init_model,demo_process
from geo_city_loc_poi import init_city, load_city_npz, nearest_poi, get_latlon, get_distance, process_latlon

mean_flw_magn = 0
p1 = 0
@njit(fastmath=True)
def ssim_numba(img1, img2):
    # print("inside ssim_numba")
    C1 = 6.5025
    C2 = 58.5225

    mu1 = np.mean(img1)
    mu2 = np.mean(img2)

    sigma1 = np.var(img1)
    sigma2 = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2)
    # print("result: ", numerator/denominator)
    return numerator / denominator


# @njit(parallel=True, fastmath=True)
@njit(fastmath=True)
def compute_ssim_batch(frames):
    # print("inside compute_ssim_batch")
    n_frames = len(frames)
    scores = np.empty(n_frames)
    # sim_score = 0.9
    for i in prange(n_frames-1):
        # print(i)
        sim_score = ssim_numba(frames[i], frames[i + 1])
        # print("sim_score", sim_score)
        scores[i] = sim_score
    # scores[n_frames-1] = sim_score
    return scores

# @njit(fastmath=True)
def func_adapt_thres(scrs_arr):
    mean_threshold = np.mean(scrs_arr)
    min_thres = np.min(scrs_arr)
    max_thres = np.max(scrs_arr)

    std_thres = np.std(scrs_arr)
    adpt_thres = mean_threshold - (0.5 * std_thres)
    if adpt_thres >= 0.9:
        if 0.9 > min_thres > 0.7:
            adpt_thres = min_thres
        else:
            adpt_thres = 0.9
    elif adpt_thres <= 0.7:
        if 0.9 > max_thres > 0.7:
            adpt_thres = max_thres
        else:
            if 0.9 > mean_threshold > 0.7:
                adpt_thres = mean_threshold
            else:
                adpt_thres = 0.7
    return adpt_thres

# @njit(fastmath=True)
def adaptive_threshold(scores_arr,wind_len):
    idx = 0
    miss_cnt = 0
    adapt_thrs_arr_x = []
    adapt_thrs_arr_y = []
    summ_vect_arr = []
    adpt_thres = scores_arr[idx]

    indx_len = scores_arr.shape[0] - 1
    scores_arr[indx_len] = adpt_thres
    for i,sim_score in enumerate(scores_arr):
        if i > 0 and (i % wind_len ==  0 or i == indx_len):
            ssim_score_aary = scores_arr[idx:i]
            adapt_thres = func_adapt_thres(ssim_score_aary)
            adapt_thrs_arr_x.append(i + 1)
            adapt_thrs_arr_y.append(np.round(adapt_thres, 2))
            idx += wind_len
        # ssim_score_aary = np.zeros(len_thrshold)

        if sim_score <= adpt_thres:
            summ_vect_arr.append(1)
        else:
            miss_cnt += 1
            summ_vect_arr.append(0)
    return miss_cnt,adapt_thrs_arr_x,adapt_thrs_arr_y,summ_vect_arr


@njit(fastmath=True)
def edge_density_fast(gray, step=2):
    H, W = gray.shape

    edge_count = 0
    total = 0

    for i in range(1, H - 1, step):
        for j in range(1, W - 1, step):
            total += 1
            gx = (  -gray[i - 1, j - 1] + gray[i - 1, j + 1]
                    - 2 * gray[i, j - 1] + 2 * gray[i, j + 1]
                    - gray[i + 1, j - 1] + gray[i + 1, j + 1])
            gy = ( -gray[i - 1, j - 1] - 2 * gray[i - 1, j] - gray[i - 1, j + 1]
                    + gray[i + 1, j - 1] + 2 * gray[i + 1, j] + gray[i + 1, j + 1])

            mag = gx * gx + gy * gy

            if mag > 1000:
                edge_count += 1
    return edge_count / total

class ChangeDetection:
    def __init__(self):
        self.MAX_THRES = 0.9
        self.MIN_THRES = 0.7
        self.ALPHA = 0.5
        self.opt_path = "opt_path"
        self.adapt_thrs_arry = []
        self.algor = 1
        self.every = 2
        self.nFPS = 25
        self.nFrames = 1000
        self.adapt_thrs_ar_x = []
        self.missing_frames_lst = []
        self.procsd_frms_lst = []
        self.summ_vect = []
        self.TEST = False
        self.eval_acc = 0
        self.edge_threshold = 0.003
        self.ntwk = None
        self.baud = 0
        self.lat = 0
        self.lon = 0
        self.city = "Hyderabad"
        self.tree = None
        self.names = None
        self.categories = None
        logging.basicConfig(filename="log_realtime_videoprocess.txt",  # Log file name
                            filemode='a',  # Append mode
                            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.DEBUG
                            )
        logger = logging.getLogger("MyAppLogger")
        self.logger = logger

    def edge_filtering(self,sm_score,curr_gry):
        run_dbnet = False
        # ssim_score = ssim_lite(prev, curr)
        density = edge_density_numba(curr_gry)

        score = 0.6 * (1 - sm_score) + 0.4 * density

        if score > self.edge_threshold:
            run_dbnet = True
        return run_dbnet

    def contour_threshold(self,frm):
        valid_cnt = []
        total_area = 0

        contours, _ = cv.findContours(curr1_gray, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 10:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if h == 0:
                continue
            ratio = w / h
            if 0.2 < ratio < 10:
                valid_cnt.append(cnt)
                total_area += area
        H, W = bitmap.shape
        score = total_area / (H * W)
        if score > 0.003:
            run_dbnet = True
        else:
            run_dbnet = False
        return run_dbnet

    def ssim_frames(self,inp_vid,opt_frms,mdl_dtct,mdl_rcgn,nm):#, prev_lat, prev_lng):
        min_adpt = max_adpt = self.MAX_THRES
        len_thrshold = self.nFPS * self.every
        ssim_score_aary = np.zeros(len_thrshold)
        indx = 0
        cap = cv.VideoCapture(0)
        self.logger.info("capturing video started")
        tot_cnt = miss_cnt = 0
        res, prev = cap.read()
        self.nFPS = cap.get(cv.CAP_PROP_FPS)
        prev_gray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)

        res, curr = cap.read()
        curr_gray = cv.cvtColor(curr, cv.COLOR_BGR2GRAY)

        tot_cnt += 1
        self.summ_vect.append(1)

        sim_score = 0.9
        sim_score = ssim_numba(prev_gray, curr_gray)
        adapt_threshold = sim_score
        prev = curr
        prev_gray = curr_gray
        self.summ_vect.append(0)

        ssim_score_aary[indx] = sim_score
        indx += 1
        self.adapt_thrs_arry.append(np.clip(adapt_threshold, self.MIN_THRES, self.MAX_THRES))  # round 2 decimals
        self.adapt_thrs_ar_x.append(tot_cnt)
        res, curr = cap.read()  # 3 frame for 0loop continuity
        prev_txt_lst = curr_txt_lst = []
        prev_loc = nm
        curr_loc = nm
        while res is True:
            tot_cnt += 1
            curr_gray = cv.cvtColor(curr, cv.COLOR_BGR2GRAY)

            sim_score = ssim_numba(prev_gray, curr_gray)
            ssim_score_aary[indx] = sim_score
            indx += 1

            if tot_cnt%len_thrshold == 0:
                self.adapt_thrs_ar_x.append(tot_cnt+1)
                adapt_threshold = func_adapt_thres(ssim_score_aary)
                prev_gray = curr_gray
                prev = curr
                self.adapt_thrs_arry.append(np.round(adapt_threshold, 2))
                indx = 0
                ssim_score_aary = np.zeros(len_thrshold)
            if sim_score<=adapt_threshold:
                prev_gray = curr_gray
                prev = curr
                self.summ_vect.append(1)
                cv.imshow("real-time", curr)
                curr_txt_lst = set(demo_process(curr, curr_gray, mdl_dtct, mdl_rcgn))
                curr_lat,curr_lon,_ = get_latlon(self.ntwk,self.baud)
                loc_txt, mvmnt = process_latlon(self.city,self.lat,self.lon,curr_lat,curr_lon,self.tree,self.names,self.categories)
                if mvmnt is True:
                    self.lat = curr_lat
                    self.lon = curr_lon
                    curr_loc = loc_txt

            else:
                miss_cnt += 1
                self.summ_vect.append(0)
            res, curr = cap.read()
            # curr_lat, curr_lng = get_latlon(ntwk,9600)
            if not curr_txt_lst.issubset(prev_txt_lst):
                prev_txt_lst = curr_txt_lst
                self.logger.info(f"frame{tot_cnt},TxtLst{curr_txt_lst}")
            if prev_loc != curr_loc:
                logger.log(f"Location:{curr_loc}")
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()
        self.logger.info("capturing video ended")
        return miss_cnt

    def video_to_frames(self,inp_vid,dbnet_dtct,crnn_recgn,name):#,p_lat,p_lng):
        miss_frm_cnt = 0
        opt_pth = 'results'
        miss_frm_cnt = self.ssim_frames(inp_vid, opt_pth, dbnet_dtct, crnn_recgn, name)#, p_lat, p_lng)
        return miss_frm_cnt, opt_pth


if __name__ == '__main__':
    lst_indx = 1
    cdobj = ChangeDetection()
    inp_fldr = os.getcwd()
    csv_path = os.path.join(inp_fldr,'record_tvsumm_OptcFlw_2sec_FstMth_f1_acc.csv')
    mdl,crnn = init_model()
    city_name,(nwk,bud,lt,ln) = init_city()
    cdobj.ntwk = nwk
    cdobj.baud = bud
    cdobj.lat = lt
    cdobj.lon = ln
    # city_name = "Hyderabad, Telengana, India"
    # city_name = None
    # city_name = city_details.split(',')[0]
    print(city_name)
    if city_name != "INDOOR": #None:

        rdns, tree, thrs,wghts, nms, ctgrs = load_city_npz(city_name)
        cdobj.tree = tree
        cdobj.names = nms
        cdobj.categories = ctgrs
        name, category, dist = nearest_poi(lt, ln, tree, nms, ctgrs)
        print(f'{name},{category},{dist}')
    else:
        cdobj.logger.error(f"unable to find the {city_name}.npz to load")
    ms_frame_cnt,op_pth = cdobj.video_to_frames('real-time',mdl,crnn,name)#,lt,ln)

    lst_indx += 1
    if cdobj.eval_acc == 50:
        print("SUCCESS")