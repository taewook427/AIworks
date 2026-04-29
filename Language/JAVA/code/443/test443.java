import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Container;
import java.awt.Graphics;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;

import javax.swing.JFrame;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JPanel;

public class test443 extends JFrame{
    static int[][] inImage = new int[512][512];
    static int[][] outImage = new int[512][512];

    Container contentPane;

    public static void main(String[] args) throws Exception{
        loadImage();

        new test443();
    }

    static public void loadImage() throws Exception{
        int i, k;

        File inFp;
        FileInputStream inFs;
        inFp = new File("C:\\Users\\taewo\\Desktop\\프로그램\\JAVA\\라이트\\443\\test443.raw");

        inFs = new FileInputStream( inFp.getPath() );

        for (i = 0; i < 512; i++){
            for (k = 0; k < 512; k++){
                inImage[i][k] = inFs.read();
                outImage[i][k] = inImage[i][k];
            }
        }

        inFs.close();
    }

    test443(){
        setTitle("사진 처리 프로그램");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        contentPane = getContentPane();

        addMenu();

        DrawImage panel = new DrawImage();
        contentPane.add(panel, BorderLayout.CENTER);

        setSize(8 + 512 + 8, 25 + 31 + 512 + 8);
        setVisible(true);

        displayImage();
    }

    class DrawImage extends JPanel{

        public void paintComponent(Graphics g){
            super.paintComponent(g);

            int R, G, B;
            int i, k;
            for (i = 0; i < 512; i++){
                for (k = 0; k < 512; k++){
                    R = G = B = (int) outImage[i][k];
                    g.setColor( new Color(R, G, B) );
                    g.drawRect(k, i, 1, 1);
                }
            }
        }
    }

    void displayImage(){
        Graphics g = contentPane.getGraphics();
        contentPane.paintAll(g);
    }

    public void addMenu(){
        JMenuBar menuBar = new JMenuBar();
        setJMenuBar(menuBar);
        JMenu photoMenu = new JMenu("사진 처리");
        menuBar.add(photoMenu);

        JMenuItem equalAction = new JMenuItem("동일한 사진");
        JMenuItem negativeAction = new JMenuItem("반전된 사진");
        JMenuItem mirror1Action = new JMenuItem("좌우 대칭 사진");
        JMenuItem mirror2Action = new JMenuItem("상하 대칭 사진");
        JMenuItem saveAction = new JMenuItem("저장");
        JMenuItem exitAction = new JMenuItem("Exit");

        photoMenu.add(equalAction);
        photoMenu.add(negativeAction);
        photoMenu.add(mirror1Action);
        photoMenu.add(mirror2Action);
        photoMenu.add(saveAction);
        photoMenu.add(exitAction);

        equalAction.addActionListener( new ActionListener() {
            public void actionPerformed(ActionEvent arg0){
                equal();
            }
        } );

        negativeAction.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent arg0){
                negative();
            }
        });

        mirror1Action.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent arg0){
                mirror1();
            }
        });

        mirror2Action.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent arg0){
                mirror2();
            }
        });

        saveAction.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent arg0){
                saveImage();
            }
        });

        exitAction.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent arg0){
                System.exit(0);
            }
        });
    }

    void equal() {
        int i, k;
        for (i = 0; i < 512; i++){
            for (k = 0; k < 512; k++){
                outImage[i][k] = inImage[i][k];
            }
        }
        displayImage();
    }

    void negative() {
        int i, k;
        for (i = 0; i < 512; i++){
            for (k = 0; k < 512; k++){
                outImage[i][k] = 255 - inImage[i][k];
            }
        }
        displayImage();
    }

    void mirror1() {
        int i, k;
        for (i = 0; i < 512; i++){
            for (k = 0; k < 512; k++){
                outImage[i][k] = inImage[i][511 - k];
            }
        }
        displayImage();
    }

    void mirror2() {
        int i, k;
        for (i = 0; i < 512; i++){
            for (k = 0; k < 512; k++){
                outImage[i][k] = inImage[511 - i][k];
            }
        }
        displayImage();
    }

    void saveImage(){
        int i, k;
        String newFname = "C:\\Users\\taewo\\Desktop\\프로그램\\JAVA\\라이트\\443\\new.raw";
        File outFp;
        FileOutputStream outFs;
        outFp = new File(newFname);
        try{
            outFs = new FileOutputStream( outFp.getPath() );
            for (i = 0; i < 512; i++){
                for (k = 0; k < 512; k++){
                    outFs.write( outImage[i][k] );
                }
            }
            outFs.close();
            JOptionPane.showMessageDialog(null, "파일 저장 성공",
            "파일 저장", JOptionPane.INFORMATION_MESSAGE);
        } catch (Exception e){
            e.printStackTrace();
        }
    }
}