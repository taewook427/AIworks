import java.awt.*;
import javax.swing.*;

public class test431b{
    static class mygui extends JFrame{
        mygui() {
            setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            setTitle("test431");

            this.setLayout( new FlowLayout() );

            ImageIcon img0 = new ImageIcon("./431/p0.jpg"); // 이미지 파일 열기
            ImageIcon img1 = new ImageIcon("./431/p1.png"); // 이미지 못열면 그냥 점프하는 듯

            JButton btn0 = new JButton("버튼 0", img0); // 이미지가 있는 버튼
            this.add(btn0);

            JLabel lbl0 = new JLabel("라벨 0 입니다."); // 텍스트 라벨
            JLabel lbl1 = new JLabel(img1); // 이미지가 있는 라벨
            this.add(lbl0);
            this.add(lbl1);

            JCheckBox chk0 = new JCheckBox("C++"); // 체크박스
            JCheckBox chk1 = new JCheckBox("Java");
            JCheckBox chk2 = new JCheckBox("Python", true); // 기본적으로 체크되어 있는 체크박스
            this.add(chk0);
            this.add(chk1);
            this.add(chk2);

            JRadioButton rdo0 = new JRadioButton("고래"); // 라디오버튼
            JRadioButton rdo1 = new JRadioButton("상어");
            JRadioButton rdo2 = new JRadioButton("새우");
            this.add(rdo0);
            this.add(rdo1);
            this.add(rdo2);

            ButtonGroup grp0 = new ButtonGroup(); // 라디오버튼용 버튼그룹
            grp0.add(rdo0);
            grp0.add(rdo1);
            grp0.add(rdo2);

            setSize(1000, 1000);
            setVisible(true);
        }
    }
    public static void main(String[] args){
        System.out.println( System.getProperty("user.dir") ); // 현재 실행 폴더는 사용자 폴더로 잡는 듯 하다
        new mygui();
    }
}