public class test421{
    public static void main(String[] args){
        int[] a = new int[5]; // float b[] = new float[3]; 도 가능
        byte[] c; // byte c[]; 도 가능
        c = new byte[7]; // 7개 크기의 배열 생성

        a[0] = 1;
        a[1] = 2;
        a[2] = 3;
        a[3] = 4;
        a[4] = 5;
        System.out.println(a[0] + a[1] + a[2] + a[3] + a[4]);

        int[] b = {100, 200, 300, 400}; // 선언과 함께 초기화
        System.out.printf("배열 길이 : %d 배열 크기 : %d\n", b.length, b.length * Integer.BYTES); // 배열의 길이

        int[][] d = new int[4][5]; // 4행 5열 다차원 배열
        int i = 0;
        int j = 0;
        for (i = 0; i < 4; i++){
            for (j = 0; j < 5; j++){
                d[i][j] = i + j;
            }
        }
        for (i = 0; i < 4; i++){
            System.out.printf("%d %d %d %d %d\n", d[i][0], d[i][1], d[i][2], d[i][3], d[i][4]);
        }
        int[][] e = { {0,1,2,3,4}, {5,6,7,8,9}, {10,11,12,13,14}, {15,16,17,18,19} }; // 다차원 배열 초기화

        char[] stack = new char[5]; // 크기 5 stack
        int top = 0;

        stack[top] = 'A'; // 들어온 역순으로 나감
        top = top + 1;

        stack[top] = 'B';
        top = top + 1;

        stack[top] = 'C';
        top = top + 1;
        
        top = top - 1;
        System.out.println(stack[top]);

        top = top - 1;
        System.out.println(stack[top]);

        top = top - 1;
        System.out.println(stack[top]);

        char[] queue = {0,0,0,0,0}; // 크기 5 queue
        int rear = 0;

        queue[rear] = 'A'; // 들어온 순서대로 나감
        rear = rear + 1;

        queue[rear] = 'B';
        rear = rear + 1;

        queue[rear] = 'C';
        rear = rear + 1;

        System.out.println(queue[0]);
        for (i = 0; i < 4; i++){
            queue[i] = queue[i+1];
        }
        
        System.out.println(queue[0]);
        for (i = 0; i < 4; i++){
            queue[i] = queue[i+1];
        }

        System.out.println(queue[0]);
        for (i = 0; i < 4; i++){
            queue[i] = queue[i+1];
        }
    }
}