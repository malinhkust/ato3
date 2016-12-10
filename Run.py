import sys
sys.path.append('Libs')
sys.path.append('Src')

from path import Path
import cv2
from image_indentifier import ImageIndentifer





def pickTrueElem (QueryList, TorF_list):
    N = len(TorF_list)
    assert N == len(QueryList)

    resultList = [ QueryList[i] for i in range(N) if TorF_list[i]]

    return resultList

def pickFalseElem (QueryList, TorF_list):
    N = len(TorF_list)
    assert N == len(QueryList)

    resultList = [ QueryList[i] for i in range(N) if not TorF_list[i]]

    return resultList

if __name__ == '__main__':
    ExpID = 1

    if ExpID == 1:
        DataPath = Path('/home/malin/MLCH/ATO_projects/Ato_player3/DataImages/1481117891.2_Data/CargoData')
        image_path_list = DataPath.files('*.png')
        AP_idf = ImageIndentifer('Res/AP_res')
        CR_idf = ImageIndentifer('Res/CargoRoute_res')
        image_list = map(cv2.imread, image_path_list)
        print 'load data finished!'
        print len(image_list)

        IsAP_list = map(AP_idf.judge, image_list)
        print 'AP...done!'
        print IsAP_list

        AP_image_list = pickTrueElem(image_list, IsAP_list)

        cv2.namedWindow('ap',cv2.WINDOW_AUTOSIZE)
        for img in AP_image_list:
            cv2.imshow('ap',img)
            cv2.waitKey()
        cv2.destroyAllWindows()

        IsCR_list = map(CR_idf.judge, image_list)
        print 'CR...done!'
        print IsCR_list
        CR_image_list = pickTrueElem(image_list, IsCR_list)
        cv2.namedWindow('cr',cv2.WINDOW_AUTOSIZE)
        for img in CR_image_list:
            cv2.imshow('cr',img)
            cv2.waitKey()
        cv2.destroyAllWindows()

    if ExpID == 2: # test pick ture function
        A = [1,2,3]
        B = [True,False,True]
        print pickTrueElem(A,B)
        # print pickTrue2(A,B)/
