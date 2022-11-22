import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.GridLayout;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Vector;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfInt4;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;
import org.opencv.imgproc.Imgproc;
public class traitementImage {
	public static Runnable startVideoShapeDetection(JPanel videoFeed,
            JPanel processedFeed1,
            VideoCapture video) {
		return () -> {
			final Mat frame = new Mat();
			
			
			while (video.read(frame)) {
			
				List<MatOfPoint> contours = traitementImage.DetecterContours(frame,false);
				Mat processed3=traitementImage.detectionv2(frame, contours);
	
				// Draw current frame
				//traitementImage.drawImage(frame, videoFeed);
				traitementImage.drawImage(processed3, videoFeed);
			}
		};
	}
	
	public static void versionVideo(String name) {
		traitementImage.initiate();
		VideoCapture video = new VideoCapture(name);
		JPanel videoFeed = new JPanel();
		JPanel contourFeed= new JPanel();
		traitementImage.createJFrame(videoFeed);
		
		//video.set(Highgui.CV_CAP_PROP_FRAME_HEIGHT,480);
		//video.set(Highgui.CV_CAP_PROP_FRAME_WIDTH,640);
		
		traitementImage.startVideoShapeDetection(videoFeed, contourFeed, video).run();
	}
	
	public static void versionWebcam(int n) {
		traitementImage.initiate();
		
		final JPanel cameraFeed = new JPanel();
        final JPanel HSVFeed= new JPanel();
        final JPanel seuillageFeed= new JPanel();
        final JPanel contourFeed= new JPanel();
        traitementImage.createJFrame(contourFeed);
        
        final VideoCapture camera = new VideoCapture(n); // 1 pour caméra externe 0 pour webcam du pc
        
        traitementImage.startShapeDetection(cameraFeed, HSVFeed, seuillageFeed, contourFeed, camera).run();
	}
	
	public static Runnable startShapeDetection(final JPanel cameraFeed,
            final JPanel processedFeed1,
            final JPanel processedFeed2,
            final JPanel processedFeed3,
            final VideoCapture camera) {
		return () -> {
			final Mat frame = new Mat();
			
			while (true) {
			// Read frame from camera
			camera.read(frame);
			// Process frame
			//final Mat processed1 = traitementImage.HSV(frame);
			//final Mat processed2 = traitementImage.seuillagev2(frame);
			
			List<MatOfPoint> contours = traitementImage.DetecterContours(frame,false);
			final Mat processed3=traitementImage.detectionv2(frame, contours);
			
			
			// Draw current frame
			//traitementImage.drawImage(frame, cameraFeed);
			//traitementImage.drawImage(processed1, processedFeed1);
			//traitementImage.drawImage(processed2, processedFeed2);
			traitementImage.drawImage(processed3, processedFeed3);
			}
		};
	}
	
	public static Mat detectionv2(Mat m, List<MatOfPoint> contours) {
		
		String[] panneaux={"ref30.jpg","ref50.jpg","ref70.jpg","ref90.jpg","ref110.jpg","refdouble.jpg"};
		
		float score[]=new float[panneaux.length];
		float nbrMatch[]=new float[panneaux.length];
		Mat ball=new Mat();
		MatOfPoint2f matOfPoint2f=new MatOfPoint2f();
		float[] radius=new float[1];
		Point center=new Point();
		Mat m1=Mat.zeros(m.size(),m.type());
		m.copyTo(m1);
		
		for(int c=0;c<contours.size();c++) {
			MatOfPoint contour=contours.get(c); // chaque contour doit representr une colonne
			double contourArea=Imgproc.contourArea(contour);
			matOfPoint2f.fromList(contour.toList());
			Imgproc.minEnclosingCircle(matOfPoint2f, center, radius); // calcule le cercle minimal entourant l'objet
			if((contourArea/(Math.PI*radius[0]*radius[0]))>=0.8){ // 0.8 c'est bien 
				// si le rapport aire contour sur aire d'un cercle est bon alors c'est un cercle
				Core.circle(m1, center, (int) radius[0], new Scalar(0, 255, 0), 2); // on trace le cercle
				Rect rect=Imgproc.boundingRect(contour); // on détecte et on trace un rectangle autour
				Core.rectangle(m1,  new Point(rect.x, rect.y), new Point(rect.x+rect.width, rect.y+rect.height), new Scalar(0, 255, 0), 2);
				Mat tmp=m.submat(rect.y, rect.y+rect.height, rect.x, rect.x+rect.width); // on crée une sous matrice avec le rectangle uniquement
				ball=Mat.zeros(tmp.size(), tmp.type());
				tmp.copyTo(ball);
				
				for(int i=0;i<panneaux.length;i++) {
					// Mise à l'échelle
					Mat sroadSign = LectureImage(panneaux[i]);
					Mat sObject = new Mat();
					Imgproc.resize(ball, sObject, sroadSign.size()); // resize ball dans sObject à la taille indiquée
					Mat grayObject = new Mat(sObject.rows(),sObject.cols(),sObject.type());
					Imgproc.cvtColor(sObject, grayObject, Imgproc.COLOR_BGRA2GRAY);
					Core.normalize(grayObject, grayObject,0,255,Core.NORM_MINMAX);
					Mat graySign = new Mat(sroadSign.rows(),sroadSign.cols(),sroadSign.type());
					Imgproc.cvtColor(sroadSign, graySign, Imgproc.COLOR_BGRA2GRAY);
					Core.normalize(graySign, graySign,0,255,Core.NORM_MINMAX);
					
					//Extraction des descripteurs et keypoints
					FeatureDetector orbDetector = FeatureDetector.create(FeatureDetector.ORB);
					DescriptorExtractor orbExtractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
					
					MatOfKeyPoint objectKeypoints = new MatOfKeyPoint();
					orbDetector.detect(grayObject, objectKeypoints);
					
					MatOfKeyPoint signKeypoints = new MatOfKeyPoint();
					orbDetector.detect(graySign, signKeypoints);
					
					Mat objectDescriptor = new Mat(ball.rows(),ball.cols(),ball.type());
					orbExtractor.compute(grayObject, objectKeypoints, objectDescriptor);
					Mat signDescriptor = new Mat(sroadSign.rows(),sroadSign.cols(),sroadSign.type());
					orbExtractor.compute(graySign, signKeypoints, signDescriptor);

					// Faire le matching
					MatOfDMatch matchs = new MatOfDMatch();
					DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE);
					matcher.match(objectDescriptor, signDescriptor,matchs);
					Mat matchedImage = new Mat(sroadSign.rows(),sroadSign.cols()*2,sroadSign.type());
					Features2d.drawMatches(sObject, objectKeypoints, sroadSign, signKeypoints, matchs, matchedImage);
					
					List<DMatch> liste=new ArrayList<DMatch>();
					liste=matchs.toList();
					for(int j=0;j<liste.size();j++) {
						score[i]=score[i]+liste.get(j).distance;
					}
					
					nbrMatch[i]=matchs.rows();
					
					
				}
				int l=min(score);
				System.out.println("panneau détecté : "+panneaux[l]);
			}
			
		}
		
		return m1;
		
		
	}
	
	public static Mat seuillagev2(Mat hsv_image) {
		Mat threshold_img=DetecterCercles(hsv_image); // on va selectionner sur l'image que les couleurs choisies
		
		return threshold_img;
	}
	
	public static Mat processImage(final Mat mat) {
        final Mat processed = new Mat(mat.height(), mat.width(), mat.type());
        // Blur an image using a Gaussian filter
        Imgproc.GaussianBlur(mat, processed, new Size(7, 7), 1);

        // Switch from RGB to GRAY
        Imgproc.cvtColor(processed, processed, Imgproc.COLOR_RGB2GRAY);

        // Find edges in an image using the Canny algorithm
        Imgproc.Canny(processed, processed, 200, 25);

        // Dilate an image by using a specific structuring element
        // https://en.wikipedia.org/wiki/Dilation_(morphology)
        Imgproc.dilate(processed, processed, new Mat(), new Point(-1, -1), 1);

        return processed;
    }
	
	public static void createJFrame(final JPanel... panels) {
        final JFrame window = new JFrame("Shape Detection");
        window.setSize(new Dimension(panels.length * 640, panels.length* 480));
        window.setLocationRelativeTo(null);
        window.setResizable(true);
        window.setLayout(new GridLayout(1, panels.length));
        for (final JPanel panel : panels) {
            window.add(panel);
        }

        window.setVisible(true);
        window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }
	
	public static void drawImage(final Mat mat, final JPanel panel) {
        // Get buffered image from mat frame
        final BufferedImage image = convertMatToBufferedImage(mat);
        //final Image newImage=image.getScaledInstance(640, 480, Image.SCALE_DEFAULT);
        // Draw image to panel
        final Graphics graphics = panel.getGraphics();
        graphics.drawImage(image, 0, 0, panel);
    }
    
    private static BufferedImage convertMatToBufferedImage(final Mat mat) {
        // Create buffered image
        final BufferedImage bufferedImage = new BufferedImage(
                mat.width(),
                mat.height(),
                mat.channels() == 1 ? BufferedImage.TYPE_BYTE_GRAY : BufferedImage.TYPE_3BYTE_BGR
        );

        // Write data to image
        final WritableRaster raster = bufferedImage.getRaster();
        final DataBufferByte dataBuffer = (DataBufferByte) raster.getDataBuffer();
        mat.get(0, 0, dataBuffer.getData());

        return bufferedImage;
    }
	
	public static Mat seuillage(String name,boolean aff) {
		Mat hsv_image=traitementImage.BGR2HSV(name,false);
		Mat threshold_img=DetecterCercles(hsv_image); // on va selectionner sur l'image que les couleurs choisies
		if(aff) {
			ImShow("Seuillage", threshold_img);
		}
		
		return threshold_img;
	}
	
	public static void detection(Mat m, List<MatOfPoint> contours,boolean aff) {
		
		String[] panneaux={"ref30.jpg","ref50.jpg","ref70.jpg","ref90.jpg","ref110.jpg","refdouble.jpg"};
		//String[] panneaux={"Ball_three.png"};
		
		//ArrayList score=new ArrayList();
		float score[]=new float[panneaux.length];
		float nbrMatch[]=new float[panneaux.length];
		Mat ball=new Mat();
		MatOfPoint2f matOfPoint2f=new MatOfPoint2f();
		float[] radius=new float[1];
		Point center=new Point();
		Mat m1=Mat.zeros(m.size(),m.type());
		m.copyTo(m1);
		
		for(int c=0;c<contours.size();c++) {
			MatOfPoint contour=contours.get(c); // chaque contour doit representr une colonne
			double contourArea=Imgproc.contourArea(contour);
			matOfPoint2f.fromList(contour.toList());
			Imgproc.minEnclosingCircle(matOfPoint2f, center, radius); // calcule le cercle minimal entourant l'objet
			if((contourArea/(Math.PI*radius[0]*radius[0]))>=0.8){ // 0.8 c'est bien 
				// si le rapport aire contour sur aire d'un cercle est bon alors c'est un cercle
				Core.circle(m1, center, (int) radius[0], new Scalar(0, 255, 0), 2); // on trace le cercle
				Rect rect=Imgproc.boundingRect(contour); // on détecte et on trace un rectangle autour
				Core.rectangle(m1,  new Point(rect.x, rect.y), new Point(rect.x+rect.width, rect.y+rect.height), new Scalar(0, 255, 0), 2);
				Mat tmp=m.submat(rect.y, rect.y+rect.height, rect.x, rect.x+rect.width); // on crée une sous matrice avec le rectangle uniquement
				ball=Mat.zeros(tmp.size(), tmp.type());
				tmp.copyTo(ball);
				if(aff) {
					ImShow("Objet", ball);
				}
				
				for(int i=0;i<panneaux.length;i++) {
					// Mise à l'échelle
					Mat sroadSign = LectureImage(panneaux[i]);
					Mat sObject = new Mat();
					Imgproc.resize(ball, sObject, sroadSign.size()); // resize ball dans sObject à la taille indiquée
					Mat grayObject = new Mat(sObject.rows(),sObject.cols(),sObject.type());
					Imgproc.cvtColor(sObject, grayObject, Imgproc.COLOR_BGRA2GRAY);
					Core.normalize(grayObject, grayObject,0,255,Core.NORM_MINMAX);
					Mat graySign = new Mat(sroadSign.rows(),sroadSign.cols(),sroadSign.type());
					Imgproc.cvtColor(sroadSign, graySign, Imgproc.COLOR_BGRA2GRAY);
					Core.normalize(graySign, graySign,0,255,Core.NORM_MINMAX);
					
					//Extraction des descripteurs et keypoints
					FeatureDetector orbDetector = FeatureDetector.create(FeatureDetector.ORB);
					DescriptorExtractor orbExtractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
					
					MatOfKeyPoint objectKeypoints = new MatOfKeyPoint();
					orbDetector.detect(grayObject, objectKeypoints);
					
					MatOfKeyPoint signKeypoints = new MatOfKeyPoint();
					orbDetector.detect(graySign, signKeypoints);
					
					Mat objectDescriptor = new Mat(ball.rows(),ball.cols(),ball.type());
					orbExtractor.compute(grayObject, objectKeypoints, objectDescriptor);
					Mat signDescriptor = new Mat(sroadSign.rows(),sroadSign.cols(),sroadSign.type());
					orbExtractor.compute(graySign, signKeypoints, signDescriptor);

					// Faire le matching
					MatOfDMatch matchs = new MatOfDMatch();
					DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE);
					matcher.match(objectDescriptor, signDescriptor,matchs);
					Mat matchedImage = new Mat(sroadSign.rows(),sroadSign.cols()*2,sroadSign.type());
					Features2d.drawMatches(sObject, objectKeypoints, sroadSign, signKeypoints, matchs, matchedImage);
					
					List<DMatch> liste=new ArrayList<DMatch>();
					liste=matchs.toList();
					ArrayList matches_final = new ArrayList();
					int MAX = 40;
					
					for(int j=0;j<liste.size();j++) {
						score[i]=score[i]+liste.get(j).distance;
					}
					score[i]=score[i]/liste.size();
					
					/*
					// Autre
					for(int j=0; j<liste.size(); j++) {
					    	matches_final.add(liste.get(i).distance);
					    }
					float distance;
			  		float somme=0;
			  		int n=5;
			  		int l=matches_final.size();
			  				
			  		for(int k=0; k<l; k++) {
			  			float d=(float) matches_final.get(k);
			  		    somme=somme+d;		
			  		}
			  		
			  		distance=somme/l;
					*/
					
					nbrMatch[i]=matchs.rows();
					
					if(aff) {
					//	ImShow("matchs", matchedImage);
						//System.out.println(panneaux[i]+" nbr de match : "+matchs.rows()+"\n"+matchs.dump());
					//System.out.println(panneaux[i]+"score : "+score[i]);
						//System.out.println(liste.get(0).distance+"\n");
						//System.out.println(liste.size()+"\n");
						
					}
				}
				int l=min(score);
				System.out.println("panneau détecté : "+panneaux[l]);
			}
			
		}
		if(aff) {
			ImShow("Détection du panneau",m1);
		}
		
		
		
		
	}
	
	public static int min(float score[]) {
		float min=score[0];
		int j=0;
		for(int i=1;i<score.length;i++) {
			if(score[i]<min) {
				min=score[i];
				j=i;
			}
		}
		return j;
	}
	
	public static float max(float score[]) {
		int j=0;
		for(int i=0;i<score.length-1;i++) {
			if(score[i]>score[i+1]) {
				j=i;
			}else {
				j=i+1;
			}
		}
		return score[j];
	}
	
	public static Mat LoadImage(String name,boolean aff) {
		initiate();
		Mat m=LectureImage(name);
		if(aff) {
			ImShow("photo", m);
		}
		
		return m;
	}
	
	public static void GUI(String path) {
		SwingUtilities.invokeLater(new Runnable(){
			public void run() {
				new frame("Application Twizy",path);
			}
		});
	}
	
	public static Mat BGR2GRAY(String name,boolean aff) {
		initiate();
		Mat m=LectureImage(name);
		Mat gray_image=Mat.zeros(m.size(), m.type());
		Imgproc.cvtColor(m, gray_image, Imgproc.COLOR_BGRA2GRAY);
		if(aff) {
			ImShow("Gray", gray_image);
		}
		
		return gray_image;
	}
	
	public static Mat BGR2HSV(String name,boolean aff) {
		initiate();
		Mat m=LectureImage(name);
		Mat hsv_image=Mat.zeros(m.size(), m.type());
		Imgproc.cvtColor(m, hsv_image, Imgproc.COLOR_BGR2HSV);
		if(aff) {
			ImShow("HSV", hsv_image);
		}
		
		return hsv_image;
	}
	
	public static Mat HSV(Mat m) {
		initiate();
		Mat hsv_image=Mat.zeros(m.size(), m.type());
		Imgproc.cvtColor(m, hsv_image, Imgproc.COLOR_BGR2HSV);
		return hsv_image;
	}
	
	public static void initiate() {
		System.loadLibrary("opencv_java2413");
		System.load("C:\\Users\\zlezg\\Dropbox\\Mon PC (LAPTOP-HNBRLPHT)\\Downloads\\opencv\\build\\x64\\vc12\\bin\\opencv_ffmpeg2413_64.dll");
	}
	
	public static List<MatOfPoint> DetecterContours(Mat threshold_img,boolean aff) {
		int thresh=100;
		Mat canny_output=new Mat();
		List<MatOfPoint> contours=new ArrayList<MatOfPoint>();
		MatOfInt4 hierarchy=new MatOfInt4();
		Imgproc.Canny(threshold_img, canny_output, thresh, thresh*2);
		Imgproc.findContours(canny_output, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
		Mat drawing=Mat.zeros(canny_output.size(), CvType.CV_8UC3);
		Random rand=new Random();
		for(int i=0;i<contours.size();i++) {
			Scalar color=new Scalar(rand.nextInt(255-0+1),rand.nextInt(255-0+1),rand.nextInt(255-0+1));
			Imgproc.drawContours(drawing, contours, i, color,1,8,hierarchy,0,new Point()); // mettre color new Scalar(255,255,255)
		}
		if(aff) {
			ImShow("Contours", drawing);
		}
		
		
		return contours;
	}
	
	public static Mat DetecterCercles(Mat hsv_image) {
		Vector<Mat> channels=new Vector<Mat>();
		Core.split(hsv_image, channels);
		
		Scalar rougeorange=new Scalar(6);
		Scalar rougeviolet=new Scalar(170);
		Scalar rouge_sat=new Scalar(110);
		
		Mat m1=new Mat();
		Mat m2=new Mat();
		Mat m3=new Mat();
		Mat m4=new Mat();
		Mat m5=new Mat();
		
		Core.compare(channels.get(0), rougeorange, m1, Core.CMP_LT);
		Core.compare(channels.get(0), rougeviolet, m2, Core.CMP_GT);
		Core.compare(channels.get(1), rouge_sat, m3, Core.CMP_GT);
		Core.bitwise_or(m1, m2, m4);
		Core.bitwise_and(m4, m3, m5);
		
		return m5;
		/*
		Mat threshold_img1=new Mat();
		Mat threshold_img2=new Mat();
		Mat threshold_img=new Mat();
		Core.inRange(hsv_image, new Scalar(0, 90, 140), new Scalar(10, 255, 255), threshold_img1); // on détecte les couleurs que l'on veut
		//new Scalar(0, 90, 140), new Scalar(10, 255, 255)
		//new Scalar(0, 210, 170), new Scalar(10, 255, 230) pour p10
		//new Scalar(0, 90, 235), new Scalar(10, 180, 255) pour p1 
	 	//new Scalar(0, 130, 200), new Scalar(25, 230, 255) pour la bille 13
	 	//new Scalar(0, 200, 240), new Scalar(10, 255, 255) pour photo70
		
		Core.inRange(hsv_image, new Scalar(140, 70, 140), new Scalar(200, 255, 255), threshold_img2);
		//new Scalar(140, 70, 140), new Scalar(200, 255, 255)
		//new Scalar(140, 210, 140), new Scalar(200, 255, 200) pour p10
		//new Scalar(165, 90, 240), new Scalar(195, 180, 255) pour p1
		//new Scalar(160, 70, 240), new Scalar(190, 90, 255) pour photo70 
		
		Core.bitwise_or(threshold_img1, threshold_img2, threshold_img);
		Imgproc.GaussianBlur(threshold_img, threshold_img, new Size(9, 9), 2, 2); // Ligne utile pour le lissage des contours
		return threshold_img;
		*/
		
	}
	
	public static Mat LectureImage(String fichier) {
		File f=new File(fichier);
		Mat m=Highgui.imread(f.getAbsolutePath());
		return m;
	}
	
	public static void ImShow(String title, Mat img) {

		MatOfByte matOfByte=new MatOfByte();
		Highgui.imencode(".png", img, matOfByte);
		byte[] byteArray=matOfByte.toArray();
		BufferedImage bufImage=null;
		try {
			InputStream in=new ByteArrayInputStream(byteArray);
			bufImage=ImageIO.read(in);
			JFrame frame=new JFrame();
			frame.setTitle(title);
			frame.getContentPane().add(new JLabel(new ImageIcon(bufImage)));
			frame.pack();
			frame.setLocationRelativeTo(null);
			frame.setVisible(true);
		}catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	public static void exemples() {
		/* Exemple 0
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		Mat m=LectureImage("opencv.png");
		Mat mat=Mat.eye(3,3,CvType.CV_8UC1);
		System.out.println("mat = "+mat.dump());
		System.out.println("m = "+m.dump());
		*/
		
		
		/* Exo 1
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		Mat m=LectureImage("circles.jpg");
		for(int i=0;i<m.height();i++) {
			for(int j=0;j<m.width();j++) {
				double[] BGR=m.get(i, j);
				if(BGR[0]==255 && BGR[1]==255 && BGR[2]==255) {
					System.out.print(".");
				}else {
					System.out.print("+");
				}
			}
			System.out.println();
		}
		*/
		
		
		/* Exo 2 niv de gris
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		Mat m=LectureImage("bgr.png");
		System.out.println("m = "+m.size());
		Vector<Mat> channels=new Vector<Mat>();
		Core.split(m,channels);
		for(int i=0;i<channels.size();i++) {
			ImShow(Integer.toBinaryString(i),channels.get(i));
			//System.out.println("mat = "+channels.get(i).dump());
			System.out.println("m = "+channels.get(i).size());
		}
		
		ImShow("la vraie image",m);
		*/
		
		/* Exo 3 niv RGB
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		Mat m=LectureImage("circles.jpg");
		Vector<Mat> channels=new Vector<Mat>();
		Core.split(m,channels);
		Mat dst=Mat.zeros(m.size(), m.type());
		Vector<Mat> chans=new Vector<Mat>();
		Mat empty=Mat.zeros(m.size(),CvType.CV_8UC1);
		for(int i=0;i<channels.size();i++) {
			ImShow(Integer.toBinaryString(i),channels.get(i));
			chans.removeAllElements();
			for(int j=0;j<channels.size();j++) {
				if(j!=i) {
					chans.add(empty);
				}else {
					chans.add(channels.get(i));
				}
			}
			Core.merge(chans, dst);
			ImShow(Integer.toString(i),dst);	
		}
		ImShow("la vraie image",m);
		*/
		
		/* Exo seuillage
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		Mat m=LectureImage("circles.jpg");
		Mat hsv_image=Mat.zeros(m.size(), m.type());
		Imgproc.cvtColor(m, hsv_image, Imgproc.COLOR_BGR2HSV);
		Mat threshold_img1=new Mat();
		Mat threshold_img2=new Mat();
		Mat threshold_img=new Mat();
		Core.inRange(hsv_image, new Scalar(0, 100, 100), new Scalar(10, 255, 255), threshold_img1);
		Core.inRange(hsv_image, new Scalar(160, 100, 100), new Scalar(179, 255, 255), threshold_img2);
		Core.bitwise_or(threshold_img1, threshold_img2, threshold_img);
		//Imgproc.GaussianBlur(threshold_img, threshold_img, new Size(9, 9), 2, 2); // Ligne utile pour le lissage des contours  
		ImShow("Cercles rouge", threshold_img);
		ImShow("la vraie image",m);
		*/
		
		/* Exo extraire les contours des cercles 
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		Mat m=LectureImage("Billard_Balls.jpg");
		ImShow("Cercles", m);
		Mat hsv_image=Mat.zeros(m.size(), m.type());
		Imgproc.cvtColor(m, hsv_image, Imgproc.COLOR_BGR2HSV);
		ImShow("HSV", hsv_image);
		Mat threshold_img=DetecterCercles(hsv_image);
		ImShow("Seuillage", threshold_img);
		int thresh=100;
		Mat canny_output=new Mat();
		List<MatOfPoint> contours=new ArrayList<MatOfPoint>();
		MatOfInt4 hierarchy=new MatOfInt4();
		Imgproc.Canny(threshold_img, canny_output, thresh, thresh*2);
		Imgproc.findContours(canny_output, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
		Mat drawing=Mat.zeros(canny_output.size(), CvType.CV_8UC3);
		Random rand=new Random();
		for(int i=0;i<contours.size();i++) {
			Scalar color=new Scalar(rand.nextInt(255-0+1),rand.nextInt(255-0+1),rand.nextInt(255-0+1));
			Imgproc.drawContours(drawing, contours, i, color,1,8,hierarchy,0,new Point());
		}
		ImShow("Contours", drawing);
		*/
		
		/* Exo reconnaissance des cercles rouges
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		Mat m=LectureImage("circles_rectangles.jpg");
		ImShow("Cercles", m);
		Mat hsv_image=Mat.zeros(m.size(), m.type());
		Imgproc.cvtColor(m, hsv_image, Imgproc.COLOR_BGR2HSV);
		ImShow("HSV", hsv_image);
		Mat threshold_img=DetecterCercles(hsv_image);
		ImShow("Seuillage", threshold_img);
		List<MatOfPoint> contours=DetecterContours(threshold_img, false);
		
		MatOfPoint2f matOfPoint2f=new MatOfPoint2f();
		float[] radius=new float[1];
		Point center=new Point();
		for(int c=0;c<contours.size();c++) {
			MatOfPoint contour=contours.get(c);
			double contourArea=Imgproc.contourArea(contour);
			matOfPoint2f.fromList(contour.toList());
			Imgproc.minEnclosingCircle(matOfPoint2f,  center,  radius);
			if((contourArea/(Math.PI*radius[0]*radius[0]))>=0.8) {
				Core.circle(m, center, (int) radius[0], new Scalar(0, 255, 0), 2);
			}
		}
		ImShow("Détection des cercles rouges",m);
		*/
		
		/*
		Core.circle(m, center, (int) radius[0], new Scalar(0, 255, 0), 2); // on le trace
		Rect rect=Imgproc.boundingRect(contour); // on détecte et on trace un rectangle autour
		Core.rectangle(m,  new Point(rect.x, rect.y), new Point(rect.x+rect.width, rect.y+rect.height), new Scalar(0, 255, 0), 2);
		Mat tmp=m.submat(rect.y, rect.y+rect.height, rect.x, rect.x+rect.width); // on crée une sous matrice avec le rectangle uniquement
		ball=Mat.zeros(tmp.size(), tmp.type());
		tmp.copyTo(ball);
		ImShow("Ball", ball);
		
		// Mise à l'échelle
		Mat sroadSign = LectureImage("ref70.jpg");
		Mat sObject = new Mat();
		Imgproc.resize(ball, sObject, sroadSign.size()); // resize ball dans sObject à la taille indiquée
		Mat grayObject = new Mat(sObject.rows(),sObject.cols(),sObject.type());
		BGR2GRAY(sObject, grayObject);
		//ImShow("objetc gray",grayObject);
		Core.normalize(grayObject, grayObject,0,255,Core.NORM_MINMAX);
		Mat graySign = new Mat(sroadSign.rows(),sroadSign.cols(),sroadSign.type());
		BGR2GRAY(sroadSign, graySign);
		//ImShow("roadsign gray",graySign);
		Core.normalize(graySign, graySign,0,255,Core.NORM_MINMAX);
		
		//Extraction des descripteurs et keypoints
		FeatureDetector orbDetector = FeatureDetector.create(FeatureDetector.ORB);
		DescriptorExtractor orbExtractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
		
		MatOfKeyPoint objectKeypoints = new MatOfKeyPoint();
		orbDetector.detect(grayObject, objectKeypoints);
		
		MatOfKeyPoint signKeypoints = new MatOfKeyPoint();
		orbDetector.detect(graySign, signKeypoints);
		
		Mat objectDescriptor = new Mat(ball.rows(),ball.cols(),ball.type());
		orbExtractor.compute(grayObject, objectKeypoints, objectDescriptor);
		
		Mat signDescriptor = new Mat(sroadSign.rows(),sroadSign.cols(),sroadSign.type());
		orbExtractor.compute(graySign, signKeypoints, signDescriptor);
		
		// Faire le matching
		MatOfDMatch matchs = new MatOfDMatch();
		DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE);
		matcher.match(objectDescriptor, signDescriptor,matchs);
		System.out.println(matchs.dump());
		Mat matchedImage = new Mat(sroadSign.rows(),sroadSign.cols()*2,sroadSign.type());
		Features2d.drawMatches(sObject, objectKeypoints, sroadSign, signKeypoints, matchs, matchedImage);
		
		ImShow("matchs", matchedImage);
		*/
	}
}
