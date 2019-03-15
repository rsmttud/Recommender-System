import json


class LabelMap:
    """
    A class which represents a label map. It loads a json - label map, where keys are label names. The method provides
    two methods to get the label name as string and one to get the label names as int.
    """
    def __init__(self, path_to_json: str):
        self.label_map = {}
        self.__initialize_map(path_to_json)
        self.labels = self.label_map.keys()

    def __repr__(self):
        return str(self.label_map)

    def __initialize_map(self, path_to_json):
        """
        Initializes the map by building the mapping dict.

        :param path_to_json: Path where the json with label information is stored
        :type path_to_json: str

        :return: None
        """
        file = open(path_to_json, "r")
        for dict in json.load(file)["labels"]:
            self.label_map.update({dict["name"]: dict["id"]})

    def get_name(self, label_id: int) -> str:
        """
        Returns the label name as string for a given integer.

        :param label_id: label id
        :type label_id: int

        :return: the text label
        :rtype: str
        """
        for class_name, _id in self.label_map.items():
            if _id == label_id:
                return class_name

    def get_index(self, class_name: str) -> int:
        """
        Gives the the label_id as integer given a label name as string

        :param class_name: label name
        :type class_name: str

        :return: label id
        :rtype: int
        """
        return self.label_map[class_name]
