public class test439c{
    public static void main(String[] args){
        int arr[] = {9, 3, 7, 2, 5};
        int maxtemp, mintemp;
        int i = 0;

        for (i = 0; i < 5; i++){
            int j = 0;
            for (j = 0; j < 4; j++){
                maxtemp = Math.max( arr[j], arr[j+1] );
                mintemp = Math.min( arr[j], arr[j+1] );
                arr[j] = mintemp;
                arr[j+1] = maxtemp;
            }
        }

        for (i = 0; i < 5; i++){
            System.out.printf("%d ", arr[i]);
        }
    }
}

// Math 클래스
// abs(숫자) : 절댓값
// ceil(double), floor(double) : 올림, 내림
// sin(double), cos(double), tan(double) : 삼각비
// max(숫자, 숫자), min(숫자, 숫자) : 최대, 최소
// pow(double, double) : 앞 숫자 ** 뒷 숫자
// random() : 0.0 이상 1.0 미만 double 랜덤값
// round(숫자) : double -> long, float -> int 반올림
// sqrt(double) : 제곱근

// 기본 데이터 형식 - 래퍼 클래스
// byte - Byte
// char - Character
// short - Short
// int - Integer
// long - Long
// float - Float
// double - Double
// 기본데이터형식 var = 래퍼클래스.valueOf(문자열)
// int a = 100; / Integer b = 100;

// 원시형 vs 객체형 - 매서드 사용 가능 여부
// Integer.toString(a); vs a.toString();

// 하지만 자바9 부터 구식화되어 사용이 금지된 듯...