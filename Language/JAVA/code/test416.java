import java.io.IOException;
import java.util.Scanner;

public class test416{
    public static void main(String[] args) throws IOException{
        int a, b;
        int result;
        char k;

        Scanner s = new Scanner(System.in);
        System.out.print("첫 번째 계산할 정수 : ");
        a = s.nextInt();

        System.out.print("중간 연산자( + - * / % ) : ");
        k = (char) System.in.read();

        System.out.print("두 번째 계산할 정수 : ");
        b = s.nextInt();

        if (k == '+') {
            result = a + b;
            System.out.println(a + " + " + b + " = " + result);
        }

        if (k == '-') {
            result = a - b;
            System.out.println(a + " - " + b + " = " + result);
        }

        if (k == '*') {
            result = a * b;
            System.out.println(a + " * " + b + " = " + result);
        }

        if (k == '/') {
            if (b != 0) {
                result = a / b;
                System.out.println(a + " / " + b + " = " + result);
                float result2;
                result2 = b;
                result2 = a / result2;
                System.out.println(a + " / " + b + " = " + result2);
            }
            else {
                System.out.println("0으로 나눌 수 없습니다!");
            }
        }

        if (k == '%') {
            if (b != 0) {
                result = a % b;
                System.out.println(a + " % " + b + " = " + result);
            }
            else {
                System.out.println("0으로 나눈 나머지를 구할 수 없습니다!");
            }
        }

    }
}