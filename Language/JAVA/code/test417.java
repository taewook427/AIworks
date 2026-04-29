public class test417{
    public static void main(String[] args){
        System.out.print("Hello, world!");
        System.out.println("Hello, world!");
        System.out.printf("%d\n", 100 + 100);
        System.out.printf("%d + %d = %d\n", 100, 50, 150);
        System.out.printf("%d %x %o %f %c %s\n", 256, 0x45A, 0177, 3.1415, 'A', "Hello");
        System.out.printf("%5d\n%05d\n%7.1f\n%7.3f\n%10s\n", 123, 123, 123.45, 123.45, "Hi JAVA");
        
        int a = 100;
        long b = 2147483648L;
        float c = 123.45f;
        double d = 3.1415926535;
        System.out.printf("%d %d %f %f\n", a, b, c, d);
        a = (int) 123.45;
        c = (float) 200;
        System.out.printf("%d %f\n", a, c);

        a = 100;
        b = a; //값만 복사해서 전달
        b = b + 50;
        System.out.printf("%d %d\n", a, b);

        char e = 'A'; // char e = 65; 와 같다
        String f = "Hello";
        System.out.printf("%c %d %s %%\n", e, (int)e, f);

        byte g = 127; // -128 ~ 127
        short h = 32767; // -32768 ~ 32767
        int i = 2147483647; // 32bit
        long j = 2147483648L; // 64bit
        float k = 123.45f; //유효숫자 6자리
        double l = 3.1415926535; //유효숫자 14자리
        System.out.printf("%d %d %d %d %f %.15f\n", g, h, i, j, k, l);
        
        boolean m = true, n = false;
        n = (10 == 20);
        System.out.printf("%s %s\n", m, n);

        System.out.printf("%d %d %d %d %d %d %d\n", Byte.SIZE, Short.SIZE, Integer.SIZE, Long.SIZE, Float.SIZE, Double.SIZE, Character.SIZE);
        String o = "Hello, world! 안녕";
        for (i = 1; i < o.length() + 1; i++){
            System.out.print( o.charAt( o.length() - i ) );
        }
    }
}