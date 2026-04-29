public class test422{
    public static void main(String[] args){
        String a = "Hello, world!\n안녕, 자바!"; // 문자열 선언
        int b;
        b = a.length(); // 문자열 길이
        int i;
        for (i = 0; i < b; i++){
            System.out.printf( "%c", a.charAt(i) + 1); //  문자열 한 글자
        }
        System.out.printf( "\n%s %s\n", a.startsWith("H"), a.endsWith(".") ); // 시작과 끝 비교

        a = "Java를 공부하는 중. Java는 옛날 언어에요.";
        System.out.printf( "%d %d\n", a.indexOf("Java"), a.lastIndexOf("Java") ); // 문자열 찾기
        System.out.println( a.replace("Java", "자바") ); // 문자열 치환
        System.out.println( a.substring(0,4) ); // 문자열 스플라이싱
        String c[] = a.split(" "); // 문자열 쪼개기
        for (i = 0; i < c.length; i++){
            System.out.println( c[i] );
        }
        c = a.split("\\."); // 매개변수가 정규 표현식을 입력받는다. "[.]" 도 가능
        for (i = 0; i < c.length; i++){
            System.out.println( c[i] );
        }
        System.out.println( a.toLowerCase() ); //소문자로
        System.out.println( a.toUpperCase() ); //대문자로
        a = "     a     b     c     ";
        System.out.println( a.trim() ); //앞뒤만 공백제거
        String txt0 = "Java Programming";
        String txt1 = "Java Programming!";
        System.out.println( txt0.compareTo(txt1) ); //문자열 동일 확인, 결과가 0이 아니면 false, 0 이면 true
        System.out.println( txt0.contains("Java") ); //문자열 포함 확인, 결과는 true/false
        txt0 = "Java Programming"; //메모리 주소 -> 1234
        txt1 = "Java Programming"; //메모리 주소 -> 1234
        String txt2 = new String("Java Programming"); //메모리 주소 -> 3456
        System.out.println( txt0 == txt1 ); // 값 + 주소 비교
        System.out.println( txt0 == txt2 ); // 값 + 주소 비교
        System.out.println( txt0.equals(txt1) ); // 값 비교
        System.out.println( txt0.equals(txt2) ); // 값 비교
        System.out.println( txt0.compareTo(txt2) ); // 값 비교
    }
}