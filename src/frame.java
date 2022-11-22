import java.awt.BorderLayout;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.List;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
public class frame extends JFrame implements ActionListener{
private static final long serialVersionUID=1L;
	
	JButton buttonHSV;
	JButton buttonImage;
	JButton buttonSeuillage;
	JButton buttonContours;
	JButton buttonDetection;
	JLabel label;
	JPanel panel;
	String name;
	
	public frame(String title,String n) {
		name=n;
		buttonHSV=new JButton("HSV");
		buttonContours=new JButton("Contours");
		buttonContours.setActionCommand("Contours");
		buttonDetection=new JButton("Detection");
		buttonDetection.setActionCommand("Detection");
		buttonSeuillage=new JButton("Seuillage");
		buttonSeuillage.setActionCommand("Seuillage");
		buttonImage=new JButton("Image");
		buttonImage.setActionCommand("Image");
		buttonHSV.setActionCommand("HSV");
		label=new JLabel("Traitement d'image");
		panel=new JPanel();
		buttonContours.addActionListener(this);
		buttonDetection.addActionListener(this);
		buttonHSV.addActionListener(this);
		buttonImage.addActionListener(this);
		buttonSeuillage.addActionListener(this);
		panel.setBorder(BorderFactory.createEmptyBorder(30, 30, 10, 30));
		panel.setLayout(new GridLayout(0, 1));
		panel.add(buttonImage);
		panel.add(buttonHSV);
		panel.add(buttonSeuillage);
		panel.add(buttonContours);
		panel.add(buttonDetection);
		panel.add(label);
		add(panel,BorderLayout.CENTER);		
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		
		setTitle(title);
		pack();
		//setSize(1280,720);
		setLocationRelativeTo(null);
		setVisible(true);
	}
	
	

	



	@Override
	public void actionPerformed(ActionEvent e) {
				
		Mat m=traitementImage.LoadImage(name,false);
		//Mat mGray=traitementImage.BGR2GRAY(name,false);
		Mat hsv_image=traitementImage.BGR2HSV(name,false);
		Mat threshold_img=traitementImage.seuillage(name,false);
		List<MatOfPoint> contours=traitementImage.DetecterContours(threshold_img,false); // détection des contours sur les couleurs choisies uniquement
		
		
		String command=e.getActionCommand();
		
		if(command.equals("HSV")) {
			traitementImage.ImShow("HSV", hsv_image); 
		}else if(command.equals("Image")) {
			traitementImage.ImShow("photo", m); 
		}else if(command.equals("Seuillage")) {
			traitementImage.ImShow("Seuillage", threshold_img); 
		}else if(command.equals("Contours")) {
			contours=traitementImage.DetecterContours(threshold_img,true);
		}else if(command.equals("Detection")) {
			traitementImage.detection(m,contours,true);
		}
	}

}
