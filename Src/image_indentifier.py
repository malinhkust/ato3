import sys
sys.path.append('../Libs')

import ato_tools as tools
import cv2
import time

###
# Target: indentify the different frame in the image .
# 1. we allow classifers cascade together.
# 2. based on image type, we call different parser.
# 3. is existing
#


ATO_FrameClass = ['AirPort','CargoRoute']

class ImageIndentifer (object):
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

    def __loadSearchParameters (self, queryImage, pyr_levels, scale_bes):
        template_list = self.Icons_dict.values()
        N = len(template_list)

        # prepare the data for map function
        scale_bes_list = tools.iter(scale_bes, N)
        pyr_levels_list = tools.iter(pyr_levels, N)
        query_image_list = tools.iter(queryImage, N)
        return query_image_list, pyr_levels_list, scale_bes_list

    def judge(self, queryImage):
        template_list = self.Icons_dict.values()
        pyr_levels = 3
        scale_bes = (1, 1.5, 5) # begin end step

        query_image_list, pyr_levels_list, scale_bes_list = \
            self.__loadSearchParameters(queryImage, pyr_levels, scale_bes)

        results_topC_rois = map(tools.searchPyramid, query_image_list, template_list, pyr_levels_list, scale_bes_list)

        score_fail_list = filter(lambda x: x[0] < 0.9, results_topC_rois)
        is_this_class = len(score_fail_list) == 0
        return is_this_class

if __name__ == '__main__':
    q = cv2.imread('/home/malin/MLCH/ATO_projects/Ato_player3/DataImages/AirPortData/1481118222.89.png')
    q_cr = cv2.imread('/home/malin/MLCH/ATO_projects/Ato_player3/DataImages/1481117891.2_Data/CargoData/1481118022.56.png')
    AP_Indentifer = ImageIndentifer('../Res/AP_res')
    CR_Indentifer = ImageIndentifer('../Res/CargoRoute_res')

    # AP_Indentifer.printAllShapes()
    # ispass = AP_Indentifer.judge(q)

    print AP_Indentifer.judge(q)
    print AP_Indentifer.judge(q_cr)
    print CR_Indentifer.judge(q)
    print CR_Indentifer.judge(q_cr)

