import cv2
import numpy as np
from path import Path
import sys
sys.path.append('../Libs')

import ato_tools as tools
import pytesseract as ocr
from PIL import Image


# import pyscreenshot as ss
# import pymouse
import time

class AirportDataDict (dict):
    def __init__(self):
        self['TravelIndex'] = 0
        self['BusinessIndex'] = 0
        self['AP_Name'] = ''
        self['AP_Idx'] = 1

    def show(self):
        for k in self.keys():
            print k, ':', self[k]


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


class RouteTabel_ColData (object):
    def __init__(self,imageList=None,bboxList=None):
        self.image_list = imageList
        self.bbox_list = bboxList
        assert len(self.image_list) == len(self.bbox_list)
        self.N = len(self.image_list)
        self.image_proc_list = []
        self.string_list = []

    def proc_images (self, proc_func):
        self.image_proc_list = map(proc_func, self.image_list)

    def ocr_images (self):
        self.string_list = map(self.ocr, self.image_proc_list)

    def ocr(self, query_images):
        cv2.imwrite('AP_name_image.png', query_images)
        resStr = ocr.image_to_string(Image.open('AP_name_image.png'), config='-psm 6')
        return resStr




class RouteListParser (object):
    def __init__(self, templateImagePath=None):
        assert isinstance(templateImagePath, str)
        self.Dir = templateImagePath
        self.Icons_dict = tools.genDictByImageAndFilenamePair(self.Dir)

    def printAllShapes(self):
        print 'icons path = ', self.Dir
        if not self.Icons_dict.keys():
            print 'Empty Icons_dict!'
        else:
            for key in self.Icons_dict.keys():
                print key, ':', self.Icons_dict[key].shape

    def __procImageForReco_APName (self,queryImage):
        AP_name_image2 = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)
        AP_name_image2 = adjust_gamma(AP_name_image2, gamma=1.5)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(3, 3))
        AP_name_image2 = clahe.apply(AP_name_image2)

        AP_name_image2 = cv2.pyrUp(AP_name_image2)
        # AP_name_image2 = cv2.pyrUp(AP_name_image2)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(3, 3))
        AP_name_image2 = clahe.apply(AP_name_image2)
        AP_name_image2 = cv2.pyrUp(AP_name_image2)
        return AP_name_image2

    def __procImageForReco_Dist (self,queryImage):
        AP_name_image2 = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)
        AP_name_image2 = adjust_gamma(AP_name_image2, gamma=1.5)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(3, 3))
        AP_name_image2 = clahe.apply(AP_name_image2)

        AP_name_image2 = cv2.pyrUp(AP_name_image2)

        return AP_name_image2

    def __genWholeColROI_xyxy (self,templateTabROI_xyxy):
        AP_NameBtn_x1 , AP_NameBtn_y1,AP_NameBtn_x2,AP_NameBtn_y2 = templateTabROI_xyxy
        dh = (AP_NameBtn_y2 - AP_NameBtn_y1)
        WholeColROI_xyxy = [AP_NameBtn_x1 , AP_NameBtn_y1+dh,AP_NameBtn_x2,AP_NameBtn_y2+8*dh]
        return WholeColROI_xyxy

    def __findRouteList_TableColunm (self,queryImage, templateImage):
        scale_StartStepStop = [1.0, 1.5, 5]
        pyLevels = 3
        top_score, templateImage_xyxy = tools.searchPyramid(QueryImage=queryImage,
                                                            template=templateImage,
                                                            pyramidLevels= pyLevels,
                                                            scaleVect_StartStopStep=scale_StartStepStop)
        assert top_score > 0.9
        WholeColROI_xyxy = self.__genWholeColROI_xyxy(templateImage_xyxy)
        WholeCol_image = tools.cutImage(tools.xyxy2xywh(WholeColROI_xyxy), queryImage)
        
        B_range , G_range, R_range = (210,255), (210,255), (210,255)
        WholeCol_image_mask = tools.threshold3(WholeCol_image, B_range, G_range, R_range)

        canvas = WholeCol_image_mask.copy()
        Total_contours, _ = cv2.findContours(canvas,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas,Total_contours,-1,(255,255,255),3)

        contour_area_list = map(cv2.contourArea,Total_contours)
        contour_area_array = np.array(contour_area_list)
        Max_Area = max(contour_area_array)

        each_col_image_list = []
        bbox_xywh_list = []

        for i, contour_i in enumerate(Total_contours):
            if contour_area_list[i] == Max_Area:
                bbox_xywh_refCoords = cv2.boundingRect(contour_i)
                bbox_xywh = list(bbox_xywh_refCoords)
                # print bbox_xywh,bbox_xywh_refCoords

                bbox_xywh[0] = WholeColROI_xyxy[0] + bbox_xywh_refCoords[0]
                bbox_xywh[1] = WholeColROI_xyxy[1] + bbox_xywh_refCoords[1]

                each_col_image = tools.cutImage(bbox_xywh_refCoords, WholeCol_image)

                each_col_image_list.append(each_col_image)
                bbox_xywh_list.append(bbox_xywh)

        return each_col_image_list,bbox_xywh_list

    def __drawColElements(self,queryImage,bbox_xywh_list):
        canvas2 = queryImage.copy()
        for bbox_xywh in bbox_xywh_list:
            canvas2 = tools.drawROI(canvas2, tools.xywh2xyxy(bbox_xywh))
        return canvas2

    def parseRouteList (self,queryImage):
        # calculate a list of candidate airport name in the region
        assert queryImage.ndim == 3
        Image_list, bbox_list = self.__findRouteList_TableColunm(queryImage,self.Icons_dict['DistanceTab'])
        DistCol = RouteTabel_ColData(Image_list, bbox_list)
        Image_list2, bbox_list2 = self.__findRouteList_TableColunm(queryImage,self.Icons_dict['TonsTab'])
        TonsCol = RouteTabel_ColData(Image_list2, bbox_list2)



        # N_Dist = len(DistImage_list)
        #
        # for i in range(N_Dist):
        #     DistImage = DistCol.image_list[i]
        #     DistImage2 = self.__procImageForReco_Dist(DistImage)
        #     cv2.imwrite('AP_name_image.png',DistImage2)
        #     DistStr = ocr.image_to_string( Image.open('AP_name_image.png'), config='-psm 6')
        #     print DistStr, ':', int(DistStr[:-2].replace(',','')) #remove km and ,




        #         # AP_name_image2 = self.__procImageForRecoAPName(AP_name_image)
        #
        #         # AP_name_image3 = cv2.equalizeHist(cv2.cvtColor(AP_name_image2,cv2.COLOR_BGR2GRAY) )
        #         # cv2.imwrite('AP_name_image.png',AP_name_image2)
        #         # cityName= ocr.image_to_string( Image.open('AP_name_image.png'),config='-psm 6')
        #         #
        #         #
        #         # if cityName[-1] == '?' or cityName[-1]=='2':
        #         #     AP_idx = 2
        #         #     cityName = cityName[:-1]
        #         #
        #         # else:
        #         #     AP_idx = 1
        #         #     cityName = cityName[:-3]
        #         #
        #         # print cityName, AP_idx
        #         bbox_xywh = list(bbox_xywh_refCoords)
        #         print bbox_xywh,bbox_xywh_refCoords
        #

        #
        canvas2 = self.__drawColElements(queryImage, DistCol.bbox_list)
        cv2.imwrite('draw_test.png',canvas2)

        return None

if __name__ == '__main__':
    DataImage_dir = '/home/malin/MLCH/ATO_projects/Ato_player3/DataImages/CargoData2'
    query_image_name = '1481126529.35.png'
    import os

    query_image_path = os.path.join(DataImage_dir, query_image_name )
    query_image = cv2.imread(query_image_path)

    Markers_path = '/home/malin/Desktop/Link to MLCH/ATO_projects/ato3/Res/CargoRouterMarkers'

    RLP = RouteListParser (Markers_path)
    RLP.printAllShapes()
    RLP.parseRouteList(query_image)