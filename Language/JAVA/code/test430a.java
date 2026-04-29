import javax.swing.JFrame;

// 레이블 구성 방식
// FlowLayout : 왼쪽 위부터 오른쪽으로 배치
// BorderLayout : 화면을 동서남북중앙 으로 나눠 배치. 레이블 위치 지정 필요.
// GridLayout : 행렬 개수대로 나누고 왼쪽 위부터 오른쪽으로 배치
// CardLayout : 화면에 버튼을 꽉차게 배치하고 뒤에 겹쳐서 배치.
// 컨테이너.setLayout( new 레이아웃종류() ); 로 사용.

public class test430a{

    static class mygui extends JFrame{ // main class 내부에 static class 생성
        mygui(){ // 생성자에 초기화 내용 포함
            setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // 닫기 버튼으로 종료
            // 이곳에 버튼 등 레이블 코딩
            setTitle("test430-gui"); // 창제목 설정
            setSize(500, 500); // 창 크기 설정
            setVisible(true); // 창이 화면에 보이게 함.
        }
    }
    public static void main(String[] args){
        new mygui(); // GUI 생성
    }
}