class Student:
    def __init__(self, id,name, age,sex, score):
        self.id = id
        self.name = name
        self.age = age
        self.score = score
        self.sex=sex
        self._score=score
    def __str__(self):
        return "id:{},name:{},age:{},sex:{},score:{}".format(self.id,self.name,self.age,self.sex,self.score)