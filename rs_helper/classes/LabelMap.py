import json


class LabelMap:

    def __init__(self, path_to_json: str):
        self.label_map = {}
        self.__initialize_map(path_to_json)
        self.labels = self.label_map.keys()

    def __repr__(self):
        return str(self.label_map)

    def __initialize_map(self, path_to_json):
        file = open(path_to_json, "r")
        for dict in json.load(file)["labels"]:
            self.label_map.update({dict["name"]: dict["id"]})

    def get_name(self, label_id: int) -> str:
        for class_name, _id in self.label_map.items():
            if _id == label_id:
                return class_name

    def get_index(self, class_name: str) -> int:
        return self.label_map[class_name]
