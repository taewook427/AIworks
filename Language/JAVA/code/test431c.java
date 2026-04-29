import java.awt.*;
import javax.swing.*;

public class test431c{
    static class mygui extends JFrame{
        mygui(){
            setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            setTitle("test431");
            this. setLayout( new FlowLayout() );
            setSize(200,400);

            JTextField txt0 = new JTextField(10); // 한줄 텍스트, 10크기
            this.add(txt0);

            JTextArea txt1 = new JTextArea(5, 10); // 5행 10열
            this.add(txt1);
            this.add( new JScrollPane(txt1) ); // 스크롤바, 5행 이상 입력시 스크롤바 표시

            JPasswordField txt2 = new JPasswordField(10); // 한줄 비밀번호 입력, 10크기, 영문만 입력
            this.add(txt2);

            String[] list0 = {"고양이", "강아지", "토끼", "기니피그", "뱀"};
            JList list1 = new JList(list0); // 리스트 표시
            this.add(list1);

            JComboBox combo0 = new JComboBox(list0); // 콤보박스 표시
            this.add(combo0);

            setVisible(true);
        }
    }
    public static void main(String[] args){
        new mygui();
    }
}