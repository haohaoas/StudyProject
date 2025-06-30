from flask import Flask, render_template, jsonify
import pymysql
import random
import datetime
import os

app = Flask(__name__)

def get_connection():
    return pymysql.connect(
        host='127.0.0.1',
        port=3306,
        user='root',
        password='153672',
        db='mydatabase',
        charset='utf8'
    )

def insert_data(name):
    conn = get_connection()
    cursor = conn.cursor()
    sql = "INSERT INTO call_name (name, call_time) VALUES (%s, %s)"
    cursor.execute(sql, (name, datetime.datetime.now()))
    conn.commit()
    cursor.close()
    conn.close()

def read_data(name):
    conn = get_connection()
    cursor = conn.cursor()
    sql = "SELECT 1 FROM call_name WHERE name = %s LIMIT 1"
    cursor.execute(sql, (name,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result is not None

def check_and_clear_database():
    conn = get_connection()
    cursor = conn.cursor()
    sql = "SELECT MIN(call_time) FROM call_name"
    cursor.execute(sql)
    result = cursor.fetchone()
    if result and result[0]:
        time_diff = datetime.datetime.now() - result[0]
        if time_diff.days >= 5:
            cursor.execute("DELETE FROM call_name")
            conn.commit()
    cursor.close()
    conn.close()

def roll_call():
    check_and_clear_database()
    if not os.path.exists('名单.txt'):
        return "名单文件不存在"

    with open('名单.txt', 'r', encoding='utf-8') as f:
        names = [line.strip() for line in f if line.strip()]
    unused = [name for name in names if not read_data(name)]

    if not unused:
        return "所有人都已被点过名"
    selected = random.choice(unused)
    insert_data(selected)
    return f"抽中姓名：{selected}"

@app.route('/')
def index():
    return  render_template('index.html')

@app.route('/rollcall')
def call():
    result = roll_call()
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
