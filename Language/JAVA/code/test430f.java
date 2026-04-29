import java.awt.*;
import javax.swing.*;

public class test430f{

    static class mygui extends JFrame{
        mygui(){
            setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            
            this.setLayout( null );
            // 레이아웃 없이 사용, 위치 수동 지정

            JButton btn0 = new JButton("버튼 0");
            btn0.setBounds(50, 50, 70, 60); // (50,50) 위치, 폭 70 높이 60
            this.add(btn0);
            JButton btn1 = new JButton("버튼 1");
            btn1.setBounds(80, 80, 70, 60);
            this.add(btn1);
            JButton btn2 = new JButton("버튼 2");
            btn2.setBounds(110, 110, 70, 60);
            this.add(btn2);

            setTitle("test430-gui");
            setSize(256, 256);
            setVisible(true);
        }
    }
    public static void main(String[] args){
        new mygui();
    }
}