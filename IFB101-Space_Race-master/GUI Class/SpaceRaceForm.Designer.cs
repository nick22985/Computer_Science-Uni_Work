namespace GUI_Class
{
    partial class SpaceRaceForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            this.splitContainer1 = new System.Windows.Forms.SplitContainer();
            this.tableLayoutPanel = new System.Windows.Forms.TableLayoutPanel();
            this.comboBox = new System.Windows.Forms.ComboBox();
            this.gamereset = new System.Windows.Forms.Button();
            this.RollDice = new System.Windows.Forms.Button();
            this.playersDataGridView = new System.Windows.Forms.DataGridView();
            this.singlestep = new System.Windows.Forms.GroupBox();
            this.SingleStepNo = new System.Windows.Forms.RadioButton();
            this.SingleStepYes = new System.Windows.Forms.RadioButton();
            this.spaceracetitle = new System.Windows.Forms.Label();
            this.numberofplayerstext = new System.Windows.Forms.Label();
            this.playerstext = new System.Windows.Forms.Label();
            this.exitButton = new System.Windows.Forms.Button();
            this.timer1 = new System.Windows.Forms.Timer(this.components);
            this.playerTokenImageDataGridViewImageColumn = new System.Windows.Forms.DataGridViewImageColumn();
            this.nameDataGridViewTextBoxColumn = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.positionDataGridViewTextBoxColumn = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.rocketFuelDataGridViewTextBoxColumn = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.playerBindingSource = new System.Windows.Forms.BindingSource(this.components);
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).BeginInit();
            this.splitContainer1.Panel1.SuspendLayout();
            this.splitContainer1.Panel2.SuspendLayout();
            this.splitContainer1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.playersDataGridView)).BeginInit();
            this.singlestep.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.playerBindingSource)).BeginInit();
            this.SuspendLayout();
            // 
            // splitContainer1
            // 
            this.splitContainer1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.splitContainer1.Location = new System.Drawing.Point(0, 0);
            this.splitContainer1.Name = "splitContainer1";
            // 
            // splitContainer1.Panel1
            // 
            this.splitContainer1.Panel1.Controls.Add(this.tableLayoutPanel);
            // 
            // splitContainer1.Panel2
            // 
            this.splitContainer1.Panel2.Controls.Add(this.comboBox);
            this.splitContainer1.Panel2.Controls.Add(this.gamereset);
            this.splitContainer1.Panel2.Controls.Add(this.RollDice);
            this.splitContainer1.Panel2.Controls.Add(this.playersDataGridView);
            this.splitContainer1.Panel2.Controls.Add(this.singlestep);
            this.splitContainer1.Panel2.Controls.Add(this.spaceracetitle);
            this.splitContainer1.Panel2.Controls.Add(this.numberofplayerstext);
            this.splitContainer1.Panel2.Controls.Add(this.playerstext);
            this.splitContainer1.Panel2.Controls.Add(this.exitButton);
            this.splitContainer1.Size = new System.Drawing.Size(884, 661);
            this.splitContainer1.SplitterDistance = 664;
            this.splitContainer1.TabIndex = 0;
            // 
            // tableLayoutPanel
            // 
            this.tableLayoutPanel.AutoSize = true;
            this.tableLayoutPanel.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.tableLayoutPanel.BackColor = System.Drawing.SystemColors.ActiveCaption;
            this.tableLayoutPanel.ColumnCount = 8;
            this.tableLayoutPanel.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 12.5F));
            this.tableLayoutPanel.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 12.5F));
            this.tableLayoutPanel.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 12.5F));
            this.tableLayoutPanel.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 12.5F));
            this.tableLayoutPanel.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 12.5F));
            this.tableLayoutPanel.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 12.5F));
            this.tableLayoutPanel.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 12.5F));
            this.tableLayoutPanel.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 12.5F));
            this.tableLayoutPanel.Dock = System.Windows.Forms.DockStyle.Fill;
            this.tableLayoutPanel.Location = new System.Drawing.Point(0, 0);
            this.tableLayoutPanel.Name = "tableLayoutPanel";
            this.tableLayoutPanel.RowCount = 7;
            this.tableLayoutPanel.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 14.28571F));
            this.tableLayoutPanel.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 14.28571F));
            this.tableLayoutPanel.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 14.28571F));
            this.tableLayoutPanel.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 14.28571F));
            this.tableLayoutPanel.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 14.28571F));
            this.tableLayoutPanel.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 14.28571F));
            this.tableLayoutPanel.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 14.28571F));
            this.tableLayoutPanel.Size = new System.Drawing.Size(664, 661);
            this.tableLayoutPanel.TabIndex = 0;
            // 
            // comboBox
            // 
            this.comboBox.FormattingEnabled = true;
            this.comboBox.Items.AddRange(new object[] {
            "2",
            "3",
            "4",
            "5",
            "6"});
            this.comboBox.Location = new System.Drawing.Point(114, 51);
            this.comboBox.Name = "comboBox";
            this.comboBox.Size = new System.Drawing.Size(35, 21);
            this.comboBox.TabIndex = 1;
            this.comboBox.Text = "6";
            this.comboBox.SelectedIndexChanged += new System.EventHandler(this.comboBox_SelectedIndexChanged);
            // 
            // gamereset
            // 
            this.gamereset.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.gamereset.Location = new System.Drawing.Point(33, 626);
            this.gamereset.Name = "gamereset";
            this.gamereset.Size = new System.Drawing.Size(75, 23);
            this.gamereset.TabIndex = 6;
            this.gamereset.Text = "Game Reset";
            this.gamereset.UseVisualStyleBackColor = true;
            this.gamereset.Click += new System.EventHandler(this.gamereset_Click);
            // 
            // RollDice
            // 
            this.RollDice.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.RollDice.Location = new System.Drawing.Point(83, 597);
            this.RollDice.Name = "RollDice";
            this.RollDice.Size = new System.Drawing.Size(75, 23);
            this.RollDice.TabIndex = 0;
            this.RollDice.Text = "Roll Dice";
            this.RollDice.UseVisualStyleBackColor = true;
            this.RollDice.Click += new System.EventHandler(this.RollDice_Click);
            // 
            // playersDataGridView
            // 
            this.playersDataGridView.AllowUserToAddRows = false;
            this.playersDataGridView.AllowUserToDeleteRows = false;
            this.playersDataGridView.AutoGenerateColumns = false;
            this.playersDataGridView.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.playersDataGridView.Columns.AddRange(new System.Windows.Forms.DataGridViewColumn[] {
            this.playerTokenImageDataGridViewImageColumn,
            this.nameDataGridViewTextBoxColumn,
            this.positionDataGridViewTextBoxColumn,
            this.rocketFuelDataGridViewTextBoxColumn});
            this.playersDataGridView.DataSource = this.playerBindingSource;
            this.playersDataGridView.Location = new System.Drawing.Point(0, 126);
            this.playersDataGridView.Name = "playersDataGridView";
            this.playersDataGridView.RowHeadersVisible = false;
            this.playersDataGridView.Size = new System.Drawing.Size(213, 196);
            this.playersDataGridView.TabIndex = 5;
            // 
            // singlestep
            // 
            this.singlestep.BackColor = System.Drawing.SystemColors.ControlDark;
            this.singlestep.Controls.Add(this.SingleStepNo);
            this.singlestep.Controls.Add(this.SingleStepYes);
            this.singlestep.Location = new System.Drawing.Point(45, 354);
            this.singlestep.Name = "singlestep";
            this.singlestep.Size = new System.Drawing.Size(140, 55);
            this.singlestep.TabIndex = 4;
            this.singlestep.TabStop = false;
            this.singlestep.Text = "Single Step?";
            // 
            // SingleStepNo
            // 
            this.SingleStepNo.AutoSize = true;
            this.SingleStepNo.Location = new System.Drawing.Point(74, 19);
            this.SingleStepNo.Name = "SingleStepNo";
            this.SingleStepNo.Size = new System.Drawing.Size(39, 17);
            this.SingleStepNo.TabIndex = 1;
            this.SingleStepNo.TabStop = true;
            this.SingleStepNo.Text = "No";
            this.SingleStepNo.UseVisualStyleBackColor = true;
            this.SingleStepNo.Click += new System.EventHandler(this.SingleStepNo_Click);
            // 
            // SingleStepYes
            // 
            this.SingleStepYes.AutoSize = true;
            this.SingleStepYes.Location = new System.Drawing.Point(25, 19);
            this.SingleStepYes.Name = "SingleStepYes";
            this.SingleStepYes.Size = new System.Drawing.Size(43, 17);
            this.SingleStepYes.TabIndex = 0;
            this.SingleStepYes.TabStop = true;
            this.SingleStepYes.Text = "Yes";
            this.SingleStepYes.UseVisualStyleBackColor = true;
            this.SingleStepYes.CheckedChanged += new System.EventHandler(this.SingleStepYes_CheckedChanged);
            this.SingleStepYes.Click += new System.EventHandler(this.SingleStepYes_Click);
            // 
            // spaceracetitle
            // 
            this.spaceracetitle.AutoSize = true;
            this.spaceracetitle.Font = new System.Drawing.Font("Microsoft Sans Serif", 15.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.spaceracetitle.Location = new System.Drawing.Point(39, 9);
            this.spaceracetitle.Name = "spaceracetitle";
            this.spaceracetitle.Size = new System.Drawing.Size(132, 25);
            this.spaceracetitle.TabIndex = 1;
            this.spaceracetitle.Text = "SpaceRace";
            // 
            // numberofplayerstext
            // 
            this.numberofplayerstext.AutoSize = true;
            this.numberofplayerstext.Location = new System.Drawing.Point(15, 51);
            this.numberofplayerstext.Name = "numberofplayerstext";
            this.numberofplayerstext.Size = new System.Drawing.Size(93, 13);
            this.numberofplayerstext.TabIndex = 2;
            this.numberofplayerstext.Text = "Number of Players";
            // 
            // playerstext
            // 
            this.playerstext.AutoSize = true;
            this.playerstext.Font = new System.Drawing.Font("Microsoft Sans Serif", 15.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.playerstext.Location = new System.Drawing.Point(65, 98);
            this.playerstext.Name = "playerstext";
            this.playerstext.Size = new System.Drawing.Size(91, 25);
            this.playerstext.TabIndex = 3;
            this.playerstext.Text = "Players";
            // 
            // exitButton
            // 
            this.exitButton.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.exitButton.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.exitButton.Location = new System.Drawing.Point(119, 626);
            this.exitButton.Name = "exitButton";
            this.exitButton.Size = new System.Drawing.Size(75, 23);
            this.exitButton.TabIndex = 0;
            this.exitButton.Text = "Exit";
            this.exitButton.UseVisualStyleBackColor = true;
            this.exitButton.Click += new System.EventHandler(this.exitButton_Click);
            // 
            // playerTokenImageDataGridViewImageColumn
            // 
            this.playerTokenImageDataGridViewImageColumn.DataPropertyName = "PlayerTokenImage";
            this.playerTokenImageDataGridViewImageColumn.HeaderText = "";
            this.playerTokenImageDataGridViewImageColumn.ImageLayout = System.Windows.Forms.DataGridViewImageCellLayout.Stretch;
            this.playerTokenImageDataGridViewImageColumn.Name = "playerTokenImageDataGridViewImageColumn";
            this.playerTokenImageDataGridViewImageColumn.ReadOnly = true;
            this.playerTokenImageDataGridViewImageColumn.Width = 20;
            // 
            // nameDataGridViewTextBoxColumn
            // 
            this.nameDataGridViewTextBoxColumn.DataPropertyName = "Name";
            this.nameDataGridViewTextBoxColumn.HeaderText = "Name";
            this.nameDataGridViewTextBoxColumn.Name = "nameDataGridViewTextBoxColumn";
            // 
            // positionDataGridViewTextBoxColumn
            // 
            this.positionDataGridViewTextBoxColumn.DataPropertyName = "Position";
            this.positionDataGridViewTextBoxColumn.HeaderText = "Square";
            this.positionDataGridViewTextBoxColumn.Name = "positionDataGridViewTextBoxColumn";
            this.positionDataGridViewTextBoxColumn.ReadOnly = true;
            this.positionDataGridViewTextBoxColumn.Width = 50;
            // 
            // rocketFuelDataGridViewTextBoxColumn
            // 
            this.rocketFuelDataGridViewTextBoxColumn.DataPropertyName = "RocketFuel";
            this.rocketFuelDataGridViewTextBoxColumn.HeaderText = "Fuel";
            this.rocketFuelDataGridViewTextBoxColumn.Name = "rocketFuelDataGridViewTextBoxColumn";
            this.rocketFuelDataGridViewTextBoxColumn.ReadOnly = true;
            this.rocketFuelDataGridViewTextBoxColumn.Width = 50;
            // 
            // playerBindingSource
            // 
            this.playerBindingSource.DataSource = typeof(Object_Classes.Player);
            // 
            // SpaceRaceForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(884, 661);
            this.Controls.Add(this.splitContainer1);
            this.Name = "SpaceRaceForm";
            this.Text = "Space Race";
            this.Load += new System.EventHandler(this.SpaceRaceForm_Load);
            this.splitContainer1.Panel1.ResumeLayout(false);
            this.splitContainer1.Panel1.PerformLayout();
            this.splitContainer1.Panel2.ResumeLayout(false);
            this.splitContainer1.Panel2.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).EndInit();
            this.splitContainer1.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.playersDataGridView)).EndInit();
            this.singlestep.ResumeLayout(false);
            this.singlestep.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.playerBindingSource)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.SplitContainer splitContainer1;
        private System.Windows.Forms.Button exitButton;
        private System.Windows.Forms.TableLayoutPanel tableLayoutPanel;
        private System.Windows.Forms.Label spaceracetitle;
        private System.Windows.Forms.Label numberofplayerstext;
        private System.Windows.Forms.Label playerstext;
        private System.Windows.Forms.GroupBox singlestep;
        private System.Windows.Forms.RadioButton SingleStepNo;
        private System.Windows.Forms.RadioButton SingleStepYes;
        private System.Windows.Forms.Button RollDice;
        private System.Windows.Forms.DataGridView playersDataGridView;
        private System.Windows.Forms.Button gamereset;
        private System.Windows.Forms.ComboBox comboBox;
        private System.Windows.Forms.BindingSource playerBindingSource;
        private System.Windows.Forms.DataGridViewImageColumn playerTokenImageDataGridViewImageColumn;
        private System.Windows.Forms.DataGridViewTextBoxColumn nameDataGridViewTextBoxColumn;
        private System.Windows.Forms.DataGridViewTextBoxColumn positionDataGridViewTextBoxColumn;
        private System.Windows.Forms.DataGridViewTextBoxColumn rocketFuelDataGridViewTextBoxColumn;
        private System.Windows.Forms.Timer timer1;
    }
}

