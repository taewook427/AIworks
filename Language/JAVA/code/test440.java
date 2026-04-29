public class test440{
    static class car0{ // static main에서 호출하려면 static class여야 하는 듯
        String car0_name;

        car0(String inpt){
            this.car0_name = inpt;
        }

        public void run0(){ // 임의의 매서드 이름
            for(int i = 0; i < 5; i++){
                try{
                    Thread.sleep(10); // 10ms 멈춤
                    System.out.println(car0_name + " 순차적으로 달립니다~");
                } catch (Exception e){

                }
            }
        }
    }

    static class car1 extends Thread{
        String car1_name;

        car1(String inpt){
            this.car1_name = inpt;
        }

        public void run(){ // start 호출 시 run 매서드가 멀티스레딩으로 실행된다
            for(int i = 0; i < 5; i++){
                try{
                    Thread.sleep(10); // 10ms 멈춤
                    System.out.println(car1_name + " 동시에 달립니다~");
                } catch (Exception e){

                }
            }
        }
    }

    static class car2{
        String car2_name;
    } // 다중 상속이 불가능할 때 멀티스레딩하기

    static class truck extends car2 implements Runnable{ // 상속 + 인터페이스 Runnable
        truck(String inpt){
            this.car2_name = inpt;
        }

        public void run(){ // Thread와 마찬가지 start 호출 시 run 매서드 멀티스레드 실행
            for(int i = 0; i < 5; i++){
                try{
                    Thread.sleep(10); // 10ms 멈춤
                    System.out.println(car2_name + " 동시에 달립니다~");
                } catch (Exception e){

                }
            }
        }
    }
    public static void main(String[] args){
        car0 c0 = new car0("자동차0");
        c0.run0();
        car0 c1 = new car0("자동차1");
        c1.run0();
        car0 c2 = new car0("자동차2");
        c2.run0();

        System.out.println("===============");

        car1 d0 = new car1("자동차A");
        d0.start(); // run 매서드 멀티스레딩 실행
        car1 d1 = new car1("자동차B");
        d1.start();
        car1 d2 = new car1("자동차C");
        d2.start();

        try{
            d0.join(); // 종료까지 대기
        } catch (Exception e){

        }

        System.out.println("===============");

        truck e0 = new truck("트럭 알파"); // 트럭 인스턴스 생성
        Thread f0 = new Thread(e0); // 스레드 인스턴스에 연결
        f0.start(); // 동시 시작
        truck e1 = new truck("트럭 베타");
        Thread f1 = new Thread(e1);
        f1.start();
        truck e2 = new truck("트럭 감마");
        Thread f2 = new Thread(e2);
        f2.start();

        //메인 스레드는 여기서 끝나지만
        // 트럭들의 스레드는 백그라운드에 남아서 계속 실행
    }
}