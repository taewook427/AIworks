public class test418{
    public static void main(String[] args){
        int a = 2, b = 3, c = 4;
        System.out.printf("%d %f\n", a * b / c, a * b / (float)c);
        // += -= *= /= %= ++ -- 존재.
        // b = a++; b = a 수행 후 a = a + 1
        // b = ++a; a = a + 1 수행 후 b = a
        // == != < > <= >= 존재.
        // && and || or ! not
        
        // & 비트 논리곱(AND) | 비트 논리함(OR) ^ 비트 배타적 논리합(XOR) ~ 비트 부정(NOT) << 왼쪽 비트시프트 >>오른쪽 비트시프트
        System.out.printf("%d %d %d\n", 10 & 7, 123 | 456, 10 ^ 7);
        byte mask = 0x0F;
        System.out.printf("%d %d\n", 65 & mask, 65 | mask);
        //마스크 &로 뒤의 4 비트만 남김, |로 앞의 4 비트만 남김
        mask = 'A' ^ 'a';
        System.out.printf("%c %c\n", mask ^ 'A', mask ^ 'a');
        int d = 192;
        System.out.printf("%d %d\n", d, ~d + 1);

        // 비트 왼쪽으로 시프트 시 전체를 왼쪽으로 한 칸 민다. 가장 왼쪽에 있던 비트는 사라지고 가장 오른쪽에 0이 채워진다.
        // 비트 오른쪽으로 시프트 시 전체를 오른쪽으로 한 칸 민다. 가장 오른쪽에 있던 비트는 사라지고 가장 왼쪽에 0이 채워진다.
        // 1010 << 0100, 0101 >> 0010, 즉 2의 거듭제곱을 곱하거나 누눈 몫이다.
        int e = 10;
        System.out.printf("%d %d %d %d\n", e<<1, e<<2, e>>1, e>>2);
        int i;
        for (i = 0; i <= 10; i++){
            byte f = 3;
            System.out.println( f << i );
            // 비트시프트는 8 비트 배수 단위로 채워짐.
        }
    }
}