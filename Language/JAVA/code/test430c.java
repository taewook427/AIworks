import java.awt.*;
import javax.swing.*;

public class test430c{

    static class mygui extends JFrame{
        mygui(){
            setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            
            this.setLayout( new BorderLayout(10, 10) ); // 수평간격, 수직간격
            // BorderLayout() 도 가능

            JButton btn0 = new JButton("버튼 0");
            this.add(btn0, BorderLayout.NORTH); // 위치 지정 필요
            JButton btn1 = new JButton("버튼 1");
            this.add(btn1, BorderLayout.WEST);
            JButton btn2 = new JButton("버튼 2");
            this.add(btn2, BorderLayout.CENTER);
            JButton btn3 = new JButton("버튼 3");
            this.add(btn3, BorderLayout.EAST);
            JButton btn4 = new JButton("버튼 4");
            this.add(btn4, BorderLayout.SOUTH);

            setTitle("test430-gui");
            setSize(256, 256);
            setVisible(true);
        }
    }
    public static void main(String[] args){
        new mygui();
    }
}