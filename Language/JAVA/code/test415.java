
import java.util.Scanner;

public class test415 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		int a, b;
		int result;
		
		Scanner s = new Scanner(System.in);
		System.out.print("첫 번째 정수 : ");
		a = s.nextInt();
		System.out.print("두 번째 정수 : ");
		b = s.nextInt();
		s.close();
		
		result = a + b;
		System.out.println(a + " + " + b + " = " + result);
		
		result = a - b;
		System.out.println(a + " - " + b + " = " + result);
		
		result = a * b;
		System.out.println(a + " * " + b + " = " + result);
		
		result = a / b;
		System.out.println(a + " / " + b + " = " + result);

	}

}
