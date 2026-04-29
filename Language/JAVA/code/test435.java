import java.awt.*;
import javax.swing.*;
import java.awt.event.*;

public class test435{
    static class mygui extends JFrame{
        mygui(){
            setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            setTitle("test435");
            this. setLayout( new FlowLayout() );
            setSize(300,300);

            JLabel label0 = new JLabel("이 글자가 바뀝니다.");
            this.add(label0);

            JMenuBar bar0 = new JMenuBar(); // 최상위 : 메뉴바
            JMenu menu0 = new JMenu("파일"); // 중간 : 메뉴
            JMenu menu1 = new JMenu("편집");

            JMenuItem item0 = new JMenuItem("새 문서"); // 최하위 : 메뉴 아이템
            JMenuItem item1 = new JMenuItem("열기");
            JMenuItem item2 = new JMenuItem("닫기");

            setJMenuBar(bar0); // 메뉴바 부착
            bar0.add(menu0); // 메뉴 부착
            bar0.add(menu1);
            menu0.add(item0); // 아이템 부착
            menu0.add(item1);
            menu0.addSeparator(); // 구분선 부착
            menu0.add(item2);

            ActionListener ac0 = new ActionListener() {
                public void actionPerformed(ActionEvent p0){
                    label0.setText("[새 문서]를 선택하셨습니다.");
                }
            };
            item0.addActionListener(ac0); // 액션과 연결

            ActionListener ac1 = new ActionListener() {
                public void actionPerformed(ActionEvent p0){
                    label0.setText("[열기]를 선택하셨습니다.");
                }
            };
            item1.addActionListener(ac1);

            ActionListener ac2 = new ActionListener() {
                public void actionPerformed(ActionEvent p0){
                    System.exit(0); // 프로그램 종료
                }
            };
            item2.addActionListener(ac2);

            JToolBar bar1 = new JToolBar();
            JButton btn0 = new JButton("새 문서");
            JButton btn1 = new JButton("열기");
            JButton btn2 = new JButton("EXIT");

            add(bar1, BorderLayout.NORTH);
            bar1.add(btn0);
            bar1.add(btn1);
            bar1.addSeparator( new Dimension(20,10) ); // 구분선 추가
            bar1.add(btn2);

            btn0.addActionListener(ac0);
            btn1.addActionListener(ac1);
            btn2.addActionListener(ac2);

            setVisible(true);
        }
    }
    public static void main(String[] args){
        new mygui();
    }
}