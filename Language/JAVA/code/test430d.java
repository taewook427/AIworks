import java.awt.*;
import javax.swing.*;

public class test430d{

    static class mygui extends JFrame{
        mygui(){
            setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            
            this.setLayout( new GridLayout(3, 3, 10, 10) ); // 기본은 1행 간격 0
            // GridLayout(), GridLayout(행, 열), GridLayout(행, 열, 수평간격, 수직간격)

            JButton[] btn = new JButton[9];
            int i;
            for (i = 0; i < 9; i++){
                btn[i] = new JButton("버튼 " + i);
                this.add( btn[i] );
            }

            setTitle("test430-gui");
            setSize(256, 256);
            setVisible(true);
        }
    }
    public static void main(String[] args){
        new mygui();
    }
}