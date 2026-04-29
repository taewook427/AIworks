import java.util.Scanner;

public class test424{
    static int coffee(int mode){ // 메소드 입출력 형식 결정
        System.out.println("1. 종이컵을 준비한다.");
        System.out.println("2. 뜨거운 물을 투입한다.");
        switch(mode){
            case 0:
                System.out.println("3. 보통커피를 탄다."); break;
            case 1:
                System.out.println("3. 블랙커피를 탄다."); break;
            case 2:
                System.out.println("3. 설탕커피를 탄다."); break;
            default:
            System.out.println("3. 아무커피를 탄다."); break;
        }
        System.out.println("4. 스푼으로 젓는다.");
        System.out.println("5. 커피를 반환한다.");
        return 0; // 반환
    }

    static int calc(int a, int b, char c){
        int output = 0;
        switch (c){
            case '+':
                output = a + b; break;
            case '-':
                output = a - b; break;
            case '*':
                output = a * b; break;
            case '/':
                output = a / b; break;
            case '%':
                output = a % b; break;
        }
        return output;
    }

    static void func0(){
        int var1 = 1; // 지역변수 선언
        System.out.println( var1 );
    }

    static void func10(int var10){
        var10 = var10 + 1;
        System.out.println( var10 );
    }

    static void func20(myInt m){
        m.value = m.value + 1;
        System.out.println( m.value );
    }

    static int var0 = 0; // 전역 변수 선언
    public static void main(String[] args){
        Scanner s = new Scanner(System.in);
        int mode = 0;
        while ( ( (mode == 0) || (mode == 1) ) || (mode == 2) ){
            System.out.print("커피 모드 선택 (0, 1, 2) : ");
            mode = s.nextInt();
            int output;
            output = coffee(mode); // 정수 반환값 저장
        }

        int a = s.nextInt();
        int b = s.nextInt();
        char c = s.next().charAt(0);
        System.out.println( calc(a, b, c) );
        s.close();

        int var1 = 2; // 지역변수 선언
        func0();
        System.out.println( var1 );
        System.out.println( var0 ); // 전역변수 호출
        int var0 = 3; // 지역변수 선언
        System.out.println( var0 ); // 지역변수 호출

        int var10 = 10;
        func10(var10); // 값에 의한 호출 (영향 안줌)
        System.out.println( var10 );

        myInt m = new myInt();
        m.value = 20;
        func10(m.value); // 값에 의한 호출 (영향 안줌)
        System.out.println( m.value ); // 기본적으로 값에 의한 호출을 한다.
        func20(m); // 참조에 의한 호출 (영향 있음)
        System.out.println( m.value ); // 배열과 클래스만 참조호출
    }
}

class myInt{
    int value;
}