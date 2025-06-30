import ast

from StudentManngerImpl import StudentManngerImpl  # 使用具体实现类
from Student import Student

if __name__ == '__main__':
    studentMannger = StudentManngerImpl()

    while True:
        print("\n1. 添加学生")
        print("2. 删除学生")
        print("3. 修改学生")
        print("4. 查找学生")
        print("5. 退出")

        code = input("请输入操作代码：")

        if code == "1":
            with open("student.txt", "r") as s:
                student_id = input("请输入学生学号：")
                found = False
                for line in s:
                    line = line.strip()
                    student_info=ast.literal_eval(line)
                    if student_info.get("id") == student_id:
                        print("该学号已存在，请重新输入")
                        found = True
                        break
                if not found:
                    name = input("请输入学生姓名：")
                    age = input("请输入学生年龄：")
                    sex = input("请输入学生性别：")
                    score = input("请输入学生成绩：")
                    student = Student(student_id, name, age, sex, score)
                    studentMannger.add_student(student)

        elif code == "2":
            student_id = input("请输入要删除的学生学号：")
            if studentMannger.search_student(student_id) is None:
                print("该学生不存在")
                continue
            studentMannger.remove_student(student_id)

        elif code == "3":
            student_id = input("请输入要修改的学生学号：")
            if studentMannger.search_student(student_id) is None:
                print("该学生不存在")
                continue
            name = input("请输入新的学生姓名：")
            age = input("请输入新的学生年龄：")
            sex = input("请输入新的学生性别：")
            score = input("请输入新的学生成绩：")
            student = Student(student_id, name, age, sex, score)
            studentMannger.update_student(student)

        elif code == "4":
            student_id = input("请输入要查找的学生学号：")
            studentMannger.search_student(student_id)

        elif code == "5":
            print("程序结束")
            break

        else:
            print("输入错误，请重新输入")
