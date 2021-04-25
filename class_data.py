from config import Hyper

class ClassData:
    def __init__(self):
        self.gt_match_dict = {} 
        self.pred_match_dict = {}
        for id in range(Hyper.num_classes):
            self.gt_match_dict[id] = []
            self.pred_match_dict[id] = []
          

    def add_matches(self, buffer, gt_match, pred_match):
        class_gt_match_dict, class_pred_match_dict = self.split_matches_from_image_into_class(buffer, gt_match, pred_match)
        for i in range(Hyper.num_classes):
            if len(class_gt_match_dict[i]) > 0:
                self.gt_match_dict[i].extend(class_gt_match_dict[i])

            if len(class_pred_match_dict[i]) > 0:
                self.pred_match_dict[i].extend(class_pred_match_dict[i])

    def split_matches_from_image_into_class(self, buffer, gt_match, pred_match):
        # Compute precision and recall at each prediction box step

        class_gt_match_dict = {} 
        class_pred_match_dict = {}
        for id in range(Hyper.num_classes):
            class_gt_match_dict[id] = []
            class_pred_match_dict[id] = []

        i = -1
        for gt_m in gt_match:
            i += 1
            gt_m = gt_m.item()
            gt_class_id = buffer.gt_class_ids[i]
            class_gt_match_dict[gt_class_id.item()].append(gt_m)

        i = -1   
        for pred_m in pred_match:
            i += 1
            pred_m = pred_m.item()
            pred_class_id = buffer.pred_class_ids[i]
            class_pred_match_dict[pred_class_id.item()].append(pred_m)
            
        return class_gt_match_dict, class_pred_match_dict
