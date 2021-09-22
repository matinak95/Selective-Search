import cv2
import os
import argparse
import numpy as np
import sys
import xml.etree.ElementTree as ET
import copy

def selective_search(img, strategy):
    """
    @brief Selective search with different strategies
    @param img The input image
    @param strategy The strategy selected ['color', 'all']
    @retval bboxes Bounding boxes
    """
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    gs = cv2.ximgproc.segmentation.createGraphSegmentation()
    ##################################################
    # TODO: For this part, please set the K as 200,  #
    #       sigma as 0.8 for the graph segmentation. #
    #       Use gs as the graph segmentation for ss  #
    #       to process after strategies are set.     #
    ##################################################
    
    gs.setK(200)
    gs.setSigma(0.8)

    ss.addImage(img)
    ss.addGraphSegmentation(gs)

    if strategy == 'color':
        curr_strategy = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor()
        ss.addStrategy(curr_strategy)
    elif strategy == 'all':
        curr_strategy = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor()
        ss.addStrategy(curr_strategy)
        curr_strategy = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyTexture()
        ss.addStrategy(curr_strategy)
        curr_strategy = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyFill()
        ss.addStrategy(curr_strategy)
        curr_strategy = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategySize()
        ss.addStrategy(curr_strategy)
    else: sys.exit(0)
        
    ##################################################
    # End of TODO                                    #
    ##################################################
    bboxes = ss.process()
    xyxy_bboxes = []

    for box in bboxes:
        x, y, w, h = box
        xyxy_bboxes.append([x, y, x+w, y + h])

    return xyxy_bboxes

def parse_annotation(anno_path):
    """
    @brief Parse annotation files for ground truth bounding boxes
    @param anno_path Path to the file
    """
    tree = ET.parse(anno_path)
    root = tree.getroot()
    gt_bboxes = []
    for child in root:
        if child.tag == 'object':
            for grandchild in child:
                if grandchild.tag == "bndbox":
                    x0 = int(grandchild.find('xmin').text)
                    x1 = int(grandchild.find('xmax').text)
                    y0 = int(grandchild.find('ymin').text)
                    y1 = int(grandchild.find('ymax').text)
                    gt_bboxes.append([x0, y0, x1, y1])
    return gt_bboxes

def bb_intersection_over_union(boxA, boxB):
    """
    @brief compute the intersaction over union (IoU) of two given bounding boxes
    @param boxA numpy array (x_min, y_min, x_max, y_max)
    @param boxB numpy array (x_min, y_min, x_max, y_max)
    """
    ##################################################
    # TODO: Implement the IoU function               #
    ##################################################
    
    x0_intersect= max(boxA[0],boxB[0])
    y0_intersect= max(boxA[1],boxB[1])

    x1_intersect= min(boxA[2],boxB[2])
    y1_intersect= min(boxA[3],boxB[3])

    if x0_intersect >= x1_intersect or y0_intersect >= y1_intersect:
        iou = 0.0
        S_intersect = 0.0
    else:
        S_intersect= (y1_intersect-y0_intersect)*(x1_intersect-x0_intersect)
        S_union = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])+(boxB[2]-boxB[0])*(boxB[3]-boxB[1])-S_intersect

        iou = S_intersect/S_union

    ##################################################
    # End of TODO                                    #
    ##################################################
    return iou, S_intersect

def visualize(img, boxes, color):
    """
    @breif Visualize boxes
    @param img The target image
    @param boxes The box list
    @param color The color
    """


    for box in boxes:
        ##################################################
        # TODO: plot the rectangles with given color in  #
        #       the img for each box.                    #
        ##################################################

        img= cv2.rectangle(img, (box[0],box[1]),(box[2],box[3]),color, 2)
    
        ##################################################
        # End of TODO                                    #
        ##################################################
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, default='color')

    args =parser.parse_args()
    img_dir = './HW2_Data/JPEGImages'
    anno_dir = './HW2_Data/Annotations'
    thres = .5

    args.strategy = input("Segmentation Strategy: 'color' or 'all'? ")
    

    

    img_list = os.listdir(img_dir)
    num_hit = 0
    num_gt = 0

    for img_path in img_list:
        """
        Load the image file here through cv2.imread
        """
        img_id = img_path[:-4]
        img_name = os.path.join(img_dir, img_path)
        ##################################################
        # TODO: Load the image with OpenCV               #
        ##################################################
        img = cv2.imread(img_name)

        ##################################################
        # End of TODO                                    #
        ##################################################

        proposals = selective_search(img, args.strategy)
        gt_bboxes = parse_annotation(os.path.join(anno_dir, img_id + ".xml"))
        iou_bboxes = []  # proposals with IoU greater than 0.5

        ##################################################
        # TODO: For all the gt_bboxes in each image,     #
        #       please calculate the recall of the       #
        #       gt_bboxes according to the document.     #
        #       Store the bboxes with IoU >= 0.5         #
        #       If there is more than one proposal has   #
        #       IoU >= 0.5 with a same groundtruth bbox, #
        #       store the one with biggest IoU.          #
        ##################################################

        S_total_gt_bb = 0.0
        S_total_intersection = 0.0
        
        for gt_bb in gt_bboxes:
            largest_iou = 0
            proposed_box= []
            S_total_gt_bb += (gt_bb[2]-gt_bb[0])*(gt_bb[3]-gt_bb[1])

            for proposal in proposals:

                curr_iou = bb_intersection_over_union(proposal, gt_bb)[0]

                if curr_iou >= max(0.5, largest_iou):
                    S_intersect_curr = bb_intersection_over_union(proposal, gt_bb)[1]
                    largest_iou = curr_iou
                    proposed_box = copy.deepcopy(proposal)

            if proposed_box: 
                iou_bboxes.append(proposed_box)
                print("Largest IoU: ",largest_iou)
                S_total_intersection += S_intersect_curr
        
        recall = S_total_intersection/S_total_gt_bb
        print("recal: ",recall)
        

        ##################################################
        # End of TODO                                    #
        ##################################################
        
        vis_img = img.copy()
        vis_img = visualize(vis_img, gt_bboxes, (255, 0, 0))
        vis_img = visualize(vis_img, iou_bboxes, (0, 0, 255))

        proposals_img = img.copy()
        proposals_img = visualize(proposals_img, gt_bboxes, (255, 0, 0))
        proposals_img = visualize(proposals_img, proposals, (0, 0, 255))

        ##################################################
        # TODO: (optional) You may use cv2 to visualize  #
        #       or save the image for report.            #
        ##################################################
        
        result_path = './Results'
        cv2.startWindowThread()
        cv2.namedWindow("preview")
        output_name = str(img_id)+"_"+str(args.strategy)+"_result.jpg"
        cv2.imwrite(os.path.join(result_path, output_name), vis_img)
        cv2.imshow("preview",vis_img); cv2.waitKey(0)
        
        cv2.startWindowThread()
        cv2.namedWindow("preview2")
        output_name = str(img_id)+"_proposal_"+str(args.strategy)+"_result.jpg"
        cv2.imwrite(os.path.join(result_path, output_name), proposals_img)
        cv2.imshow("preview2",proposals_img); cv2.waitKey(0)
        print("Number of Proposals: ",len(proposals))





        ##################################################
        # End of TODO                                    #
        ##################################################
        


if __name__ == "__main__":
    main()




