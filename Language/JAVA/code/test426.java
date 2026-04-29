import java.io.FileInputStream; // type A
import java.io.BufferedReader; // type B
import java.io.FileReader; // type B
import java.io.File; // type C
import java.util.Scanner; // type C
import java.io.FileOutputStream; // type D
import java.io.FileWriter; // type E

public class test426{
    public static void main(String[] args) throws Exception{ // FileInputStream 예외 처리 위함 type A B C D E

        // type A
        FileInputStream f = new FileInputStream("C:\\Windows\\win.ini"); // 파일 열기
        int ch = f.read(); // 없는 파일은 FileNotFoundException 발생
        while (ch != -1){ // EOF 도달 시 -1 반환
            System.out.print( (char) ch );
            ch = f.read(); // 1 바이트 읽기, 유니코드 처리 불가
        }
        f.close(); // 파일 닫기

        FileInputStream ff = new FileInputStream("C:\\Users\\taewo\\DCloud3\\settings.txt"); // 파일 열기
        byte[] chs = new byte[8192]; // 충분한 길이의 배열 설정
        ch = ff.read(); // 기본적으로 숫자(바이트)를 반환
        int count = 0;
        while (ch != -1){
            chs[count] = (byte) ch; // 1 바이트씩 읽어 배열에 추가
            ch = ff.read();
            count = count + 1;
        }
        System.out.print( new String (chs) ); // 바이트 배열을 문자열화
        ff.close(); // 파일 닫기

        // type B
        FileReader fff = new FileReader("C:\\Users\\taewo\\Desktop\\powerlog.txt"); // 파일 열기, 문자 단위로 읽음
        BufferedReader ffff = new BufferedReader(fff); // 행 단위로 읽기
        String output = ffff.readLine(); // \n을 제외한 문자열을 반환함
        while (output != null){ // 파일이 끝나면 null 반환
            System.out.println(output);
            output = ffff.readLine();
        }
        ffff.close();
        fff.close(); // 파일 닫기

        // type C
        Scanner fffff = new Scanner( new File("C:\\Users\\taewo\\DCloud3\\settings.txt") ); // 파일 열기
        while ( fffff.hasNextLine() ){ // 다음 줄을 가지면 true
            System.out.println( fffff.next() ); // next~ 사용 가능
        } // next() 는 문자열을 띄어쓰기로 구분, 공백이 나오지 않음
        fffff.close(); // 파일 닫기

        // type D
        FileOutputStream t = new FileOutputStream("C:\\Users\\taewo\\Desktop\\프로그램\\JAVA\\라이트\\test426a.txt"); //파일 열기
        // 인자 추가시 append t/f
        // 절대경로, ./ 현재 폴더 상대경로, ../ 상위폴더 상대경로
        int i = 0;
        int temp;
        for (i = 0; i < 5; i++){
            temp = System.in.read(); // Enter 누를 때 까지 계속 받기는 하는 듯
            t.write( (byte) temp ); // 1 바이트 쓰기
        } // 한글 써도 1 바이트씩 입력해서 잘 써진다.
        t.close(); // 파일 닫기

        // type E
        Scanner input = new Scanner(System.in, "EUC-KR"); // Scanner 로 한글입력
        FileWriter writer = new FileWriter("C:\\Users\\taewo\\Desktop\\프로그램\\JAVA\\라이트\\test426b.txt"); //파일 열기
        String str = input.nextLine(); // 버터 \n 처리
        str = input.nextLine(); // 입력안하고 엔터 누르면 "" 들어감.
        while ( !str.equals("") ){
            System.out.print(str + "\n");
            writer.write(str + "\n");
            str = input.nextLine();
        } // 윈도우 상에서는 EUC-KR 만 지원, 리눅스 상에서는 UTF-8 지원.
        writer.close(); // 파일 닫기
    }
}