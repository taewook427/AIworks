public class test427{
    public static void main(String[] args){
        int arglen = args.length; // length -> array, length() -> String
        int count = 0;
        System.out.println( "프로그램이 실행됩니다." );
        for (count = 0; count < arglen; count++){
            System.out.println( args[count] );
        }
    }
}