//             public - protected - default - private
// 같은 클래스    O        O            O        O
// 같은 패키지    O        O            O        X
// 하위 클래스    O        O            X        X
// 외부 클래스    O        X            X        X

class car{ // 클래스
    String color; // 클래스 변수
    int speed;

    void speedup(int value){ // 메서드
        speed = speed + value; // this 없으면 지역변수, 매개변수 우선
    }
    void speeddown(int value){
        speed = speed - value;
    }
}

class bus{
    private int speed; // 같은 클래스 내부에서만 접근 가능

    bus(){ // 생성자
        this.speed = 0; // 클래스 호출 시 처음으로 실행
    }

    public void setspeed(int value){ // 항상 접근 가능
        if ( (value >= 0) & (value <= 200) ){ // 매서드 자체적으로 speed 값의 보호가 가능
            this.speed = value; // this는 자기 클래스
        }
    }

    public int getspeed(){
        return speed;
    }
}

class taxi{
    private int speed;
    private String color;

    taxi(int value){ // 파라미터 있는 생성자
        this.speed = value;
    }

    taxi(int value, String pretty){ // 파라미터 있는 생성자
        this.speed = value;
        this.color = pretty;
    }

    public void setspeed(int value){ // 메서드 오버로딩
        if ( (value >= 0) & (value <= 200) ){
            this.speed = value;
        }
    }

    public int getspeed(){
        return speed;
    }

    public void setcolor(String value){
        this.color = value;
    }

    public String getcolor(){
        return this.color;
    }
}

class train{
    int speed; // 인스턴스 변수
    String color;
    static int people; // 클래스 변수

    static void test(){ // 클래스 매서드
        System.out.println( "함수가 실행되었습니다." );
    }
}

public class test428{
    public static void main(String[] args){
        car car0 = new car(); // 인스턴스(객체) 생성
        car car1 = new car();
        car0.color = "빨강"; // 변수 접근
        car0.speed = 0;
        car1.color = "blue";
        car1.speed = 100;
        car0.speedup(50); // 매서드 호출
        car1.speeddown(30);
        System.out.printf("%s %d, %s %d\n", car0.color, car0.speed, car1.color, car1.speed);

        bus bus0 = new bus();
        System.out.println( bus0.getspeed() );
        bus0.setspeed(80);
        bus0.setspeed(250);
        System.out.println( bus0.getspeed() );

        taxi taxi0 = new taxi(120);
        System.out.println( taxi0.getspeed() );
        taxi taxi1 = new taxi(120, "red");
        System.out.printf( "%d %s\n", taxi1.getspeed(), taxi1.getcolor() );

        train train0 = new train();
        train0.speed = 80; // 인스턴스 변수는 각 클래스마다 따로
        train0.people = 25; // 클래스 변수는 공유 (static)
        train train1 = new train();
        train1.speed = 90;
        System.out.printf("%d %d, %d %d\n", train0.speed, train0.people, train1.speed, train1.people);
        train.test(); // 클래스 매서드는 객체 생선 전에도 사용 가능 (static)
    }
}