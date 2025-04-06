class Class2:
    def __init__(self):
        self.name = "Class2"
        self.class1_instance = None
        self.class3_instance = None

    def set_class1(self, class1_instance):
        self.class1_instance = class1_instance

    def get_name(self):
        return self.name