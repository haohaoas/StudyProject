from  abc import ABC,abstractmethod


class StudentManager(ABC):
    def __init__(self):
        self.students=[]
    @abstractmethod
    def add_student(self, student):
       pass
    @abstractmethod
    def remove_student(self, id):
        pass
    @abstractmethod
    def search_student(self, name):
       pass
    @abstractmethod
    def update_student(self, student):
        pass


