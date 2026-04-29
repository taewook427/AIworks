import java.awt.*;
import javax.swing.*;
import java.awt.event.*;

public class test434{
    static class mygui extends JFrame{
        mygui(){
            setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            setTitle("test434");
            this. setLayout( new FlowLayout() );
            setSize(200,600);

            JButton btn0 = new JButton("버튼 0");
            btn0.addActionListener( new ActionListener() { // 리스너는 인터페이스라 메서드로 확장해야 한다.
                public void actionPerformed(ActionEvent arg0){ // 함수명 고정
                    btn0.setBackground(Color.RED);
                }
            } );
            this.add(btn0);

            // ActionListener는 인터페이스, 다음 형태 예시.
            // interface ActionListener { public void actionPerformed(ActionEvent e); }

            // ActionListener act0 = new ActionListener(){ public void actionPerformed(ActionEvente p0){...} };
            // btn0.addActionListener(act); 와 같이 써도 됨.

            // 이벤트 종류
            // ActionEvent : 
            // JButton - 마우스/키보드로 누름
            // JMenuItem - 메뉴 아이템 선택
            // JList - 리스트 아이템 선택
            // JTextField - Enter 누름
            // KeyEvent : 
            // 모든 컴포넌트 - 키보드 누름
            // MouseEvent : 
            // 모든 컴포넌트 - 마우스 누름/뗌/클릭/커서올림/커서내림
            // TextEvent : 
            // JTextField/JTextArea/JPasswordField - 텍스트 변경

            JTextField txtX = new JTextField(10);
            this.add(txtX);
            JTextField txtY = new JTextField(10);
            this.add(txtY);

            MouseListener ms = new MouseListener() {
                public void mouseClicked(MouseEvent e){ // 마우스 클릭
                    txtX.setText( Integer.toString( e.getX() ) );
                    txtY.setText( Integer.toString( e.getY() ) );
                }
                public void mouseEntered(MouseEvent e){}
                public void mouseExited(MouseEvent e){}
                public void mousePressed(MouseEvent e){}
                public void mouseReleased(MouseEvent e){} // 인터페이스의 4개 메서드를 필수 구현해야 한다.
            };
            this.addMouseListener(ms);

            JTextField txt0 = new JTextField(10);
            this.add(txt0);
            JTextArea txt1 = new JTextArea(10, 10);
            this.add(txt1);

            KeyAdapter ka = new KeyAdapter() {
                public void keyReleased(KeyEvent p0){
                    int key0 = p0.getKeyCode(); // 눌린 키 정수값 저장

                    if (key0 == KeyEvent.VK_ENTER){
                        String str0 = txt0.getText(); // 텍스트 입력창에서 가져오기
                        txt1.setText(txt1.getText() + str0 + '\n');
                        txt0.setText("");
                    }

                    if ( !(key0 >= KeyEvent.VK_0 && key0 <= KeyEvent.VK_9) ){
                        String str0 = txt0.getText();
                        int len0 = str0.length();
                        if (len0 != 0){
                            txt0.setText( str0.substring(0, len0 - 1) );
                        }
                    }
                }
            };
            txt0.addKeyListener(ka);

            setVisible(true);
        }
    }
    public static void main(String[] args){
        new mygui();
    }
}