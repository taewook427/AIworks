class car{ // 공통 내용을 가진 상위 클래스
    protected int speed;
    protected String color;

    car(){
        System.out.println("생성자 0번 호출");
    }

    public void setspeed(int value){
        this.speed = value;
    }

    public int getspeed(){
        return this.speed;
    }

    public void setcolor(String value){
        this.color = value;
    }

    public String getcolor(){
        return this.color;
    }
}

class sedan extends car{ // 하위 클래스
    private int people;

    sedan(){
        System.out.println("생성자 1번 호출"); // 자동으로 상위 클래스 생성자 호출 후 호출됨
    }

    public void setspeed(int value){ // 매서드 오버라이딩
        if ( (value >= 0) & (value <= 120) ){
            this.speed = value;
        }
    }

    public void setpeople(int value){
        this.people = value;
    }

    public int getpeople(){
        return this.people;
    }
}

class upclass{
    upclass(){
        System.out.println("up출력 : 기본");
    }

    upclass(String value){
        System.out.println("up출력 : " + value);
    }

    final void dosomething(){ // 오버라이딩 불가
        System.out.println("출력을 합니다.");
    }
}

class downclass extends upclass{
    final static String cartype = "클래스2"; // 오버라이딩 불가, 변경 불가

    downclass(String value){
        System.out.println("down출력 : " + value);
    }
}

class upclass2{
    upclass2(){
        System.out.println("up출력 : 기본");
    }

    upclass2(String value){
        System.out.println("up출력 : " + value);
    }
}

class downclass2 extends upclass2{
    downclass2(String value){
        super(value); // 상위 클래스에 매개변수 전달
        System.out.println("down출력 : " + value);
    }
}

abstract class sky{ // 추상 클래스
    String color;

    abstract void dosomething(); // 추상 매서드
}

class cloud extends sky{
    void setcolor(String value){
        this.color = value;
    }

    String getcolor(){
        return this.color;
    }

    void dosomething(String value){ // 상위 클래스의 모든 추상 매서드는 반드시 상속시 구현되어야 한다
        System.out.println( "출력 : " + value );
    }
}

interface ground{ // 인터페이스, 추상 매서드와 비슷함
    static final int area = 500; // static final 변수만 사용 가능

    abstract void dosomething(); // 추상 매서드만 사용 가능, abstract는 생략 가능
}

class building implements ground{ // 상속 시에는 implements 사용
    public void dosomething(String value){ // 인터페이스의 추상 매서드 완성시엔 public 필요
        System.out.println( "출력 : " + value );
    }
}

interface top{
    static final int tall = 300;
}

class skyscrapper implements ground, top{ // 다중 상속은 추상 클래스는 허용 안됨, 인터페이스는 허용됨
    public void dosomething(String value){
        System.out.println( "출력 : " + value );
    }
}

// 일반 클래스 : 일반 필드 / static final 필드 / 일반 메서드 -> 직접 인스턴스화
// 추상 클래스 : 일반 필드 / static final 필드 / 일반 메서드 / 추상 메서드 -> 상속 후 인스턴스화
// 인터페이스 : static final 필드 / 추상 메서드 -> 상속 / 다중상속 후 인스턴스화

public class test429{
    public static void main(String[] args){
        sedan car0 = new sedan();
        car0.setspeed(100);
        car0.setcolor("빨강"); // 강위 클래스 내용 호출 가능
        car0.setpeople(4);
        System.out.printf("%d %s %d\n", car0.getspeed(), car0.getcolor(), car0.getpeople() );

        car0.setspeed(140);
        System.out.printf("%d %s %d\n", car0.getspeed(), car0.getcolor(), car0.getpeople() );

        downclass class0 = new downclass("Hello"); // 하위 클래스 생성자에만 매개변수가 들어감
        downclass2 class1 = new downclass2("Hello"); // 상위 하위 클래스 생성자 모두 매개변수가 들어감

        // 상속 제한 : private, 상속 클래스만 열람 가능 : protected
        class0.dosomething(); // 매소드 호출
        System.out.println(downclass.cartype); // 변경 불가한 변수는 클래스 호출 전에도 호출 가능

        cloud mycloud = new cloud(); // 추상 클래스는 직접 호출이 불가, 상속을 통한 호출만 가능하다
        mycloud.setcolor("하얀색");
        System.out.println( mycloud.getcolor() );
        mycloud.dosomething("Hello");

        building b63 = new building(); // 인터페이스도 상속으로만 호출 가능
        System.out.println(b63.area);
        b63.dosomething("와 높다~");

        skyscrapper dubai = new skyscrapper();
        System.out.printf("%d %d\n", dubai.area, dubai.tall);
        dubai.dosomething("높으면서 넓어요");
    }
}