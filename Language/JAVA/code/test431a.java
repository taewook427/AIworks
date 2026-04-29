import java.awt.*;
import javax.swing.*;

public class test431a{
    static class mygui extends JFrame{
        mygui() {
            setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            setTitle("test431");

            Container c = this.getContentPane(); // 배경화면을 pane 으로 표현, layout에 사용도 가능
            c.setBackground(Color.YELLOW); // 노랑색 배경

            c.setLayout( new FlowLayout() ); // layout에 사용, this.setLayout() 도 가능

            JButton btn0 = new JButton("버튼 0");
            btn0.setBackground(Color.BLACK);
            btn0.setForeground(Color.MAGENTA);
            c.add(btn0); // layout 따라 add 위치를 설정

            JButton btn1 = new JButton("버튼 1");
            btn1.setFont( new Font("맑은 고딕", Font.BOLD, 30) );
            btn1.setCursor( new Cursor(Cursor.WAIT_CURSOR) );
            btn1.setToolTipText("이 버튼은 크게 보입니다~"); // 마우스 커서가 위에 있을 때 출력되는 툴팁
            c.add(btn1);

            JButton btn2 = new JButton("버튼 2");
            btn2.setEnabled(false);
            c.add(btn2);

            setSize(256, 256);
            setVisible(true);
        }
    }
    public static void main(String[] args){
        new mygui();
    }
}

// JComponent
// -AbstractButton
// --JToggleButton
// ---JCheckBox
// ---JRadioButton
// --JButton
// -JTextComponent
// --JTextField
// ---JPasswordField
// --JTextArea
// -JLabel
// -JLayer
// -JList
// -JSlider
// -JMenu
// -JTable
// -JOptionPane
// -JScrollBar
// -JToolTip
// -JPanel
// -JToolBar
// -JComboBox

// JComponent 메서드
// void setBorder(Border), Border getBorder() : 테두리 설정
// void setBackground(Color), Color getBackground() : 배경색 설정
// void setForeground(Color), Golor getForeground() : 글자색 설정
// void setOpaque(boolean), Bollean isOpaque() : 불투명 설정
// void setFont(Font), Font getFont() : 폰트 설정
// void setCursor(Cursor), Cursor getCursor() : 마우스 커서 설정
// void setPreferredSize(Dimension), int getWidth(), int getHeight() : 크기 설정
// void setLocation(int, int), int getX(), int getY() : 위치 설정
// Point getLocationOnScreen() : 전체 화면에서 좌표 확인
// void setVisible(boolean), boolean isVisible() : 보이기 여부 설정
// void setEnable(boolean), boolean isEnable() : 활성화 여부 설정