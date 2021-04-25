

class ClassResultsBuffer:
    def __init__(self, class_name, class_id, MAP, precisions, recall):
        self.class_name = class_name
        self.class_id = class_id
        self.MAP = MAP
        self.precisions = precisions
        self.recalls = recall