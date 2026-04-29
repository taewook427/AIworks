import java.awt.*;
import javax.swing.*;

public class test430b{

    static class mygui extends JFrame{
        mygui(){
            setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            
            this.setLayout( new FlowLayout() ); // 기본은 정렬방식 FlowLayout.CENTER, 간격 5픽셀
            // FlowLayout(정렬방식), FlowLayout(정렬방식, 수평간격, 수직간격)
            // CENTER EAST WEST NORTH SOUTH

            JButton btn0 = new JButton("버튼 0");
            this.add(btn0);
            JButton btn1 = new JButton("버튼 1");
            this.add(btn1);
            JButton btn2 = new JButton("버튼 2");
            this.add(btn2);
            JButton btn3 = new JButton("버튼 3");
            this.add(btn3);
            JButton btn4 = new JButton("버튼 4");
            this.add(btn4);

            setTitle("test430-gui");
            setSize(256, 256);
            setVisible(true);
        }
    }
    public static void main(String[] args){
        new mygui();
    }
}