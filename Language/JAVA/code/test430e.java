import java.awt.*;
import javax.swing.*;

public class test430e{

    static class mygui extends JFrame{
        mygui(){
            setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            
            this.setLayout( new CardLayout(10, 10) );
            // CardLayout() 도 가능. 간격은 0, 0

            JButton btn0 = new JButton("버튼 0");
            this.add(btn0); // 맨 위 버튼만 보임
            JButton btn1 = new JButton("버튼 1");
            this.add(btn1); // 나머지는 가져져서 안보임
            JButton btn2 = new JButton("버튼 2");
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