class StudentQueryTool:
    def __init__(self, file_path):
        self.file_path = file_path
        self.students = []
        self.connect_and_load()

    def connect_and_load(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    pairs = [pair.split(':') for pair in line.split(', ')]
                    student_info = {k: v for k, v in pairs}
                    self.students.append(student_info)
        except FileNotFoundError:
            print(f"未找到文件: {self.file_path}")
        except Exception as e:
            print(f"读取文件时发生错误: {e}")

