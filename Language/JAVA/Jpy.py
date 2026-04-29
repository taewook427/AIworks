import os
import tkinter
from tkinter import filedialog

print( os.popen("cd").read() )
mode = input('0 : 자바 컴파일, 1 : 자바 실행\n')
if mode == '0':
    mainwin = tkinter.Tk()
    mainwin.title('Jpy')
    mainwin.geometry('20x20+10+10')
    mainwin.resizable(0,0)
    file = filedialog.askopenfile( title='자바 소스코드 선택', filetypes=( ('java files', '*.java'),('all files', '*.*') ) )
    file = file.name.replace('\\','/')
    file = file[file.rfind('/') + 1:]
    mainwin.destroy()
    print( os.popen(f"javac {file}").read() )
elif mode == '1':
    mainwin = tkinter.Tk()
    mainwin.title('Jpy')
    mainwin.geometry('20x20+10+10')
    mainwin.resizable(0,0)
    file = filedialog.askopenfile( title='자바 클래스파일 선택', filetypes=( ('class files', '*.class'),('all files', '*.*') ) )
    file = file.name.replace('\\','/')
    file = file[file.rfind('/') + 1:file.rfind('.')]
    mainwin.destroy()
    print( os.popen(f"java {file}").read() )
else:
    print('알맞은 명령어 찾지 못 함.')
input('press ENTER to exit...')
