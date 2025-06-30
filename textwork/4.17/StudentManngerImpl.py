import os
import ast
from StudentMannger import StudentManager
from Student import Student
class StudentManngerImpl(StudentManager):

    def add_student(self, student):
        s = open("student.txt", "a")
        student = Student(student.id, student.name, student.age, student.sex, student.score)
        # self.students.append(student.__dict__)
        s.writelines(str(student.__dict__) + '\n')
        s.close()
        print("添加成功")

    def remove_student(self, id):
        with open("student.txt", "r+", encoding="utf-8") as s:
            lines = s.readlines()
            s.seek(0)
            s.truncate()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                student = ast.literal_eval(line)
                if student.get('id') != id:
                    s.write(line + '\n')
                    print("删除成功")
                    s.close()
                    return

    def search_student(self, id):
        try:
            with open("student.txt", "r", encoding="utf-8") as s:
                for line in s:
                    line = line.strip()
                    if not line:
                        continue
                    student = ast.literal_eval(line)
                    if student.get('id') == id:
                        print("找到学生信息：", student)
                        return 1
                print("没有找到")
        except FileNotFoundError:
            print("文件未找到")
        except Exception as e:
            print(f"发生未知错误: {e}")

    import ast

    def update_student(self, student_obj):
        try:
            with open("student.txt", "r", encoding="utf-8") as s:
                lines = s.readlines()

            found = False
            updated_lines = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                student = ast.literal_eval(line)
                if student.get('id') == student_obj['id']:
                    updated_lines.append(str(student_obj) + '\n')
                    found = True
                else:
                    updated_lines.append(line + '\n')

            if found:
                with open("student.txt", "w", encoding="utf-8") as s:
                    s.writelines(updated_lines)
                print("修改成功")
            else:
                print("未找到该学生，无法修改")

        except FileNotFoundError:
            print("文件未找到")
        except Exception as e:
            print(f"发生未知错误: {e}")