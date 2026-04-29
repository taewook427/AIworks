import tkinter
from tkinter import filedialog

mode = input('0 : bmp to raw, 1 : raw to bmp\n')
if mode == '0':
    mainwin = tkinter.Tk()
    mainwin.title('Jpy')
    mainwin.geometry('20x20+10+10')
    mainwin.resizable(0,0)
    file = filedialog.askopenfile( title='choose bmp', filetypes=( ('bmp files', '*.bmp'),('all files', '*.*') ) )
    file = file.name.replace('\\','/')
    mainwin.destroy()
    with open(file, 'rb') as f:
        temp = f.read()
    st = temp[10] + 256 * temp[11] + 256 * 256 * temp[12] + 256 * 256 * 256 * temp[13]
    temp = temp[st:]
    with open('test443.raw','wb') as f:
        long = 512 * 512
        temp = [ (temp[3*x] + temp[3*x+1] + temp[3*x+2])//3 for x in range(0,long) ]
        temp = list( reversed(temp) )
        for i in range(0,512):
            temp[512*i:512*i+512] = [ temp[512*i+511-x] for x in range(0,512) ]
        f.write( bytes(temp) )
    
elif mode == '1':
    mainwin = tkinter.Tk()
    mainwin.title('Jpy')
    mainwin.geometry('20x20+10+10')
    mainwin.resizable(0,0)
    file = filedialog.askopenfile( title='choose raw', filetypes=( ('raw files', '*.raw'),('all files', '*.*') ) )
    file = file.name.replace('\\','/')
    mainwin.destroy()
    header = b'BM6\x00\x0c\x00\x00\x00\x00\x006\x00\x00\x00(\x00\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x0c\x00\xc4\x0e\x00\x00\xc4\x0e\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    with open(file,'rb') as f:
        temp = list( f.read() )
    long = 512 * 512
    for i in range(0,512):
        temp[512*i:512*i+512] = [ temp[512*i+511-x] for x in range(0,512) ]
    temp = list( reversed(temp) )
    temp = [ temp[x//3] for x in range( 0,long*3) ]
    with open('new.bmp','wb') as f:
        f.write(header)
        f.write( bytes(temp) )
        
else:
    print('알맞은 명령어 찾지 못 함.')
input('press ENTER to exit...')
