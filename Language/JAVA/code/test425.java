import java.util.Scanner;
import java.io.IOException;

public class test425{
    public static void main(String[] args){
        try{ //예외 처리 구문
            int[] a = new int[3];
            a[3] = 100;
        } catch (ArrayIndexOutOfBoundsException e){ // 모든 예외를 명시
            System.out.println("배열 크기를 벗어났습니다.");
            System.out.println( e.getMessage() );
        } catch (ArithmeticException e){ // 연속 사용 가능
            System.out.println("또 다른 오류입니다.");
        }

        try{
            int b = 197 / 0;
        }catch (ArithmeticException e){
            System.out.println("0으로 나누기 오류입니다.");
        } finally{ // finally는 항상 실행
            System.out.println("항상 실행됩니다.");
        }
        // 예외는 상속 관계로 포함 관계.
        // Exception
        // - ClassNotFoundException
        // - IllegalAccessException
        // - RunTimeException
        // - - ArithmeticException
        // - - NullPointerException
        // - - IndexOutOfBoundsException
        // - - - ArrayIndexOutOfBoundsException
        // - - - StringIndexOutOfBoundsException
        // - IOException
        // - - EOFException
        // - - FileNotFoundException
        try{
            throw new Exception("이것은 오류입니다."); // 예외 던지기
        } catch (Exception e){ // 모든 종류의 예외 받기
            System.out.println( e.getMessage() );
        }

        System.out.printf("%d %c %s %x %o %f %e\n", 196, 'A', "안녕", 24, 24, 17.56f, 100.12345f); // 표준출력
        Scanner s = new Scanner(System.in);
        byte var0;
        short var1;
        int var2;
        long var3;
        float var4;
        double var5;
        String var6, var7;
        System.out.print("byte : ");
        var0 = s.nextByte(); // 초과값 입력시 오류
        System.out.print("short : ");
        var1 = s.nextShort();
        System.out.print("int : ");
        var2 = s.nextInt();
        System.out.print("long : ");
        var3 = s.nextLong();
        System.out.print("float : ");
        var4 = s.nextFloat();
        System.out.print("double : ");
        var5 = s.nextDouble();
        System.out.print("string0 : ");
        var6 = s.next(); // 입력 시 단어 뒤의 Enter 는 버퍼에 담긴다.
        System.out.print("string1 : ");
        var7 = s.nextLine(); // Buffer 의 Enter 때문에 바로 입력됨.

        String var10, var11, var12;
        System.out.print( "세 단어 입력 : " );
        var10 = s.next(); // Buffer 는 공백으로 구분되며 next는 한 버퍼의 내용만 가져온다.
        var11 = s.next();
        var12 = s.next();
        System.out.print(var10 + ", " + var11 + ", " + var12 );

        String input = "";
        int key;
        try{
            System.out.print("\nwrite : ");
            key = System.in.read(); // 한 글자 읽어서 숫자로 반환
            while (key != 13){
                input = input + Character.toString( (char) key ); // int -> char -> String
                key = System.in.read(); // 영문 1 글자 (1 바이트) 만 읽을 수 있다.
            }
        } catch (IOException e){
            e.printStackTrace();
        }
        System.out.println(input);
    }
}