import java.util.Scanner;

public class test419{
    public static void main(String[] args){
        Scanner s = new Scanner(System.in);

        int a = 99;
        if (a < 100){
            System.out.println("100 보다 작습니다.");
        }
        if (a < 50){
            System.out.println("50 보다 작습니다.");
        }
        else{
            System.out.println("50 보다 크거나 같습니다.");
        }
        if (a > 200)
            System.out.println("이 문장은 보이지 않습니다.");
        System.out.println("이 문장은 보입니다."); // 중괄호 없는 조건문은 한 문장만 담는다.

        System.out.print("1, 2, 3, 4 중 하나를 선택하세요.");
        a = s.nextInt();
        switch(a){
        case 1:
            System.out.println("1 selected");
            break;
        case 2:
            System.out.println("2 selected");
            break;
        case 3:
            System.out.println("3 selected");
            break;
        case 4:
            System.out.println("4 selected");
            break;
        default:
        System.out.println("other selected");
        } // break를 안쓰면 해당 case부터 시작해서 밑의 모든 코드를 실행함.
        System.out.print("정수를 입력하세요.");
        a = s.nextInt();
        switch(a % 4){
        case(0): System.out.println("나머지 0"); break;
        case(1): System.out.println("나머지 1"); break;
        case(2): System.out.println("나머지 2"); break;
        case(3): System.out.println("나머지 3"); break;
        } // default 없어도 됨.

        int i;
        int j;
        for (i = 1; i < 10; i++){
            for (j = 1; j <= i; j++){
                System.out.print("*");
            }
            System.out.print("\n");
        } // 1 부터 9 까지
        for (i = 0; i < 3; i++)
            System.out.print("=");
        System.out.print('\n'); // 중괄호 없는 for는 한 줄만 반복
        for (i = 9; i > 1; i--){
            for (j = 9; j > 0; j--){
                System.out.printf("%d * %d = %d\n", i, j, i * j);
            }
        }  // for 안에 조건문, 증감식은 자유로움.
        for (i = 0, j = 0; i < 3; i++, j++){
            System.out.printf("%d %d\n", i, j);
        } // 여러 증감식을 가진 for 문.
        i = 0;
        for (;i < 3;){
            System.out.println(i);
            i = i + 1;
        } // 초기화 식과 증감식은 생략 가능.
        // 조건문까지 생략한 for (;;){}은 무한루프.

        int b = 0;
        int c;
        System.out.println("어느 수까지 더할까요?");
        c = s.nextInt();
        while (c > 0){
            b += c;
            c -= 1;
        }
        System.out.println(b); // while에서 조건문만 남긴 형태가 while 문이다.
        // while (true){} 는 무한루프.
        b = 100;
        do{
            System.out.println(b);
        } while (b == 200); // do{} while ()은 처음 한 번은 실행한다.
        // break;로 현재 반복문 탈출, continue;로 탈출 후 다시 반복문으로 돌아감.
        b = 0;
        label0: for(;;){
            for (i = 1; i <= 100; i++){
                b += i;
                if (b > 2000){
                    System.out.println(b);
                    break label0; // label: for(){}, break label;로 특정 반복문 탈출 가능.
                }
            }
        }

        s.close();
        return ; // 현재 함수를 종료하고 함수를 호출한 곳으로 돌아간다. 함수 반환 자료형을 맞출 것.
    }
}