using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace WindowsFormsApplication1
{
    public partial class Form1 : Form
    {
        private delegate void SetTextDelegate(string value);
        private delegate void SafePictureDelegate(Bitmap capture);
        private Thread thread2 = null;
        private int count = 0;

        public Form1()
        {
            InitializeComponent();
            this.Size = new Size(1024, 768);
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            
        }

        private void captureBtn_click(object sender, EventArgs e)
        {
            CaptureScreen();
            
        }

        private void CaptureScreen()
        {
            while (true)
            {
                setLabelValue("Count: " + ++count);
                try
                {
                    //Creating a new Bitmap object
                    Bitmap captureBitmap = new Bitmap(1024, 768, PixelFormat.Format32bppArgb);

                    //Bitmap captureBitmap = new Bitmap(int width, int height, PixelForma
                    //Creating a Rectangle object which will  
                    //capture our Current Screen
                    Rectangle captureRectangle = Screen.AllScreens[0].Bounds;

                    //Creating a New Graphics Object
                    Graphics captureGraphics = Graphics.FromImage(captureBitmap);

                    //Copying Image from The Screen
                    captureGraphics.CopyFromScreen(captureRectangle.Left, captureRectangle.Top, 0, 0, captureRectangle.Size);
                    setPicture(captureBitmap);
                }
                catch (Exception ex)
                {
                    MessageBox.Show(ex.Message);
                }
                
            }
        }

        private void setLabelValue (string value)
        {
            if (this.label1.InvokeRequired)
            {
                SetTextDelegate d = new SetTextDelegate(setLabelValue);
                this.Invoke(d, new object[] { value });
            }
            else
            {
                label1.Text = value;
            }
        }

        private void setPicture(Bitmap capture)
        {
            if (this.label1.InvokeRequired)
            {
                SafePictureDelegate d = new SafePictureDelegate(setPicture);
                this.Invoke(d, new object[] { capture });
            }
            else
            {
                captureBox.Image = capture;
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            captureBox.Size = new Size(1024, 768);
            thread2 = new Thread(new ThreadStart(CaptureScreen));
            thread2.Start();
        }

        private void captureBox_Click(object sender, EventArgs e)
        {
            
        }

        private void label1_Click(object sender, EventArgs e)
        {

        }
    }
}
