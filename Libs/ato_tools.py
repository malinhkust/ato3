from path import Path
import cv2
import numpy as np
import cPickle

def savePickle(object, filename):
    with open(filename,'wb') as f:
        cPickle.dump(object,f)
        f.close()

def loadPickle(filename):
    with open(filename,'rb') as f:
        obj = cPickle.load(f)
        f.close()
    return obj

def bgr2gray(BGR_mat):
    if BGR_mat.ndim == 3:
        return cv2.cvtColor(BGR_mat, cv2.COLOR_BGR2GRAY)
    else:
        return  BGR_mat


def calcBetweenROI_Horizontal_xyxy(Left_xyxy, Right_xyxy, OffsetTpl_LURD=(0,0,0,0)):
    ToLeftOffset,ToUpOffset,ToRightOffset,ToDownOffset = OffsetTpl_LURD
    Bval_roi_x1 = Left_xyxy[2] - ToLeftOffset
    Bval_roi_y1 = Left_xyxy[1] - ToUpOffset
    Bval_roi_x2 = Right_xyxy[0] + ToRightOffset
    Bval_roi_y2 = Right_xyxy[3] + ToDownOffset
    btwROI_xyxy = [Bval_roi_x1,Bval_roi_y1,Bval_roi_x2,Bval_roi_y2]
    return btwROI_xyxy

def parseFilenameFromPath (inputPath):
    assert isinstance(inputPath, Path)
    dir_list = inputPath.splitall()
    filename_wExt = dir_list[-1]
    assert isinstance(filename_wExt, unicode)
    filename,ext = filename_wExt.split('.')
    return filename

def genDictByImageAndFilenamePair (inputPathOfImages):

    DH = Path(inputPathOfImages)
    if not DH.exists():
        return {}
    ImagePath_list = DH.files('*.png')
    Filename_list = map(parseFilenameFromPath, ImagePath_list)
    Image_list = map(cv2.imread, ImagePath_list)
    Icon_dict = dict (zip (Filename_list, Image_list) )
    return Icon_dict


def cutImage( roi_xywh, image):
    x, y, w, h = roi_xywh
    if image.ndim == 3:
        return image[ y:y + h,x:x + w,0:3].copy()
    return image[ y:y + h,x:x + w].copy()

def xyxy2xywh (roi_xyxy):
    x1,y1,x2,y2 = roi_xyxy
    w = x2 - x1
    h = y2 - y1
    return x1,y1,w,h

def xywh2xyxy (roi_xywh):
    x1,y1,w,h = roi_xywh
    x2 =  w + x1
    y2 = h + y1
    return x1,y1,x2,y2

def pyrDown (src, levels):
    original = src
    if levels == 0:
        return src
    for i in range(levels):
        dst = cv2.pyrDown(original)
        original = dst
    return dst

def scaleImageByFactor (img, scaleFactor):
    return cv2.resize(img, (0,0),fx=scaleFactor,fy=scaleFactor)


def matchTemplate_Scaled_XYXY (QueryImage, template,scaleVect_StartStopStep):
    Q = bgr2gray(QueryImage)
    T = bgr2gray(template)
    start, stop, step = scaleVect_StartStopStep
    # cv2.imshow('Q',Q)
    # cv2.imshow('T',T)
    # cv2.waitKey()
    top_score = 0
    best_loc = None
    Tw,Th = None, None
    assert isinstance(step,int) and step > 0
    assert stop > start
    for sf in np.linspace (start, stop, step):
        Ts= scaleImageByFactor(T,sf)
        # T_width, T_height = Ts.shape
        score_map = cv2.matchTemplate(Q,Ts,cv2.TM_CCOEFF_NORMED)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(score_map)
        # print sf,'-',maxVal
        if maxVal >= top_score:
            top_score, best_loc = maxVal, maxLoc
            Tw, Th = Ts.shape

    assert best_loc is not None
    # print best_loc[0]
    # print best_loc[1]
    startX, startY = int(best_loc[0]), int(best_loc[1])
    endX,endY = startX+Th, startY+Tw
    # print top_score, (startX,startX startY, endX, endY)
    canvas = np.dstack([Q,Q,Q])

    cv2.rectangle(canvas, (startX, startY), (endX, endY), (0, 0, 255), 4)

    # cv2.imwrite('out.png',canvas)

    # cv2.imshow("Image", T)
    # cv2.waitKey(0)
    return top_score, (startX,startY, endX, endY)



def searchPyramid (QueryImage,template,pyramidLevels,scaleVect_StartStopStep,method='TemplateMatching'):
    T_pyrd = pyrDown(template,pyramidLevels)
    Q_pyrd = pyrDown(QueryImage,pyramidLevels)
    # start, stop, step = scaleVect_StartStopStep
    # topC means best match score
    if method == 'TemplateMatching':
        topC , ROI_xyxy = matchTemplate_Scaled_XYXY(Q_pyrd,T_pyrd,scaleVect_StartStopStep)
    else:
        return ()
    scale_factor = 2 ** pyramidLevels
    FullSize_ROI_xyxy = [i * scale_factor for i in ROI_xyxy]
    return  topC, FullSize_ROI_xyxy
    # print FullSize_ROI
    # print tools.xyxy2xywh(FullSize_ROI)
    # fullsize_img = tools.cutImage( tools.xyxy2xywh(FullSize_ROI), img2 )

def drawROI (Q, roi_xyxy , color=(0,0,255), thickness=4):
    if Q.ndim !=3:
        canvas_inner = np.dstack([Q, Q, Q])
    else:
        canvas_inner = Q.copy()
    roi_xyxy_int = map(int,roi_xyxy)
    startX, startY, endX, endY = roi_xyxy_int
    # print canvas_inner.shape
    # canvas_inner = canvas.copy()
    cv2.rectangle(canvas_inner, (startX, startY), (endX, endY), (0, 0, 255), 4)
    # cv2.imwrite('canvas_inner.png', canvas_inner)
    return canvas_inner.copy()


def normalizePoint( x, y, window_xywh):
    assert isinstance(window_xywh, list) or isinstance(window_xywh, tuple)
    win_x, win_y, win_w, win_h = window_xywh
    print x, y, window_xywh
    assert win_w > 0 and win_h > 0
    delta_x = x - win_x
    delta_y = y - win_y
    assert delta_x >= 0 and delta_y >= 0
    normalized_X = float(delta_x) / float(win_w)
    normalized_Y = float(delta_y) / float(win_h)
    return normalized_X, normalized_Y


def normalizeRoi( QueryROI, window_xywh):
    x1, y1, x2, y2 = QueryROI
    x1_n, y1_n = normalizePoint(x1, y1, window_xywh)
    x2_n, y2_n = normalizePoint(x2, y2, window_xywh)
    return x1_n, y1_n, x2_n, y2_n


def calcDiagDist( roi_LeftTop_xyxy, roi_RightBottom_xyxy):
    LeftTop_x, LeftTop_y, _, _ = roi_LeftTop_xyxy
    _, _, RightBottom_x, RightBottom_y = roi_RightBottom_xyxy
    return ((LeftTop_x - RightBottom_x) ** 2 + (LeftTop_y - RightBottom_y) ** 2) ** 0.5

def threshold3 (image, R_th_MinMax_tpl, G_th_MinMax_tpl,B_th_MinMax_tpl ):
    # in treshold is True
    bgr_list = cv2.split(image)
    bgr_th_list = [B_th_MinMax_tpl,G_th_MinMax_tpl,R_th_MinMax_tpl]
    mask_ed = []

    # for (minTh, maxTh , chnl) in (bgr_th_list,bgr_list):
    for i, chnl in enumerate(bgr_list):
        minTh, maxTh = bgr_th_list[i]
        # minTh, maxTh = th_tpl
        mask = cv2.inRange(chnl,minTh,maxTh)
        # A = chnl < maxTh
        # B = chnl > minTh
        # print A.shape, A.dtype,B.shape, B.dtype
        # mask = cv2.bitwise_and(A,B)
        # print np.sum(mask)
        # mask = (chnl < maxTh) and chnl > minTh
        mask_ed.append(mask)
        # cv2.imshow('mask',mask)
        # cv2.waitKey()

    # for maski in mask_ed:
    mask3 = cv2.bitwise_and(mask_ed[0],mask_ed[1])
    mask3 = cv2.bitwise_and(mask_ed[2],mask3)

    return mask3

def calcROICenter (roi, mode):
    if mode == 'xyxy':
        roi_inner = roi
    if mode == 'xywh':
        roi_inner = xywh2xyxy(roi)
    cx = 0.5 * (roi_inner[0] + roi_inner[2])
    cy = 0.5 * (roi_inner[1] + roi_inner[3])
    return cx,cy


def scaleUpImageForOCR( queryImage):
    if queryImage.ndim==3:
        AP_name_image2 = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)
    else:
        AP_name_image2 = queryImage.copy()
    AP_name_image2 = adjust_gamma(AP_name_image2, gamma=1.5)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(3, 3))
    AP_name_image2 = clahe.apply(AP_name_image2)

    AP_name_image2 = cv2.pyrUp(AP_name_image2)
    # AP_name_image2 = cv2.pyrUp(AP_name_image2)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(3, 3))
    AP_name_image2 = clahe.apply(AP_name_image2)
    AP_name_image2 = cv2.pyrUp(AP_name_image2)
    return AP_name_image2

def iter(A, iterN):
    S = [A for i in range(iterN)]
    return S
