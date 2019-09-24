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

namespace Stardew_Fisher_DotNet
{
    public partial class Form1 : Form
    {
        private delegate void SetTextDelegate(string value);
        private delegate void SafePictureDelegate(Bitmap capture);
        private Thread thread2 = null;
        private int count = 0;
        private System.Windows.Forms.Timer timer1;

        public Form1()
        {
            InitializeComponent();
            this.Size = new Size(1024, 768);
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void timer1_Tick(object sender, EventArgs e)
        {
            count = 0;
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

        private void setLabelValue(string value)
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
                captureBox.Image.Dispose();
            }
        }

        private void startBtn_Click_1(object sender, EventArgs e)
        {
            timer1 = new System.Windows.Forms.Timer();
            timer1.Tick += new EventHandler(timer1_Tick);
            timer1.Interval = 1000; // 1 second
            timer1.Start();

            captureBox.Size = new Size(1024, 768);
            thread2 = new Thread(new ThreadStart(CaptureScreen));
            thread2.Start();
        }
    }
}
