import numpy as np
import math

def tau_Right(var1,var3):
	t2681 = -1.*var3[1]		# -p(2)
	t2690 = math.cos(var1[2])
	t2687 = math.cos(var1[3])
	t2688 = math.sin(var1[2])
	t2691 = math.sin(var1[3])
	t2684 = math.cos(var1[4])
	t2689 = t2687*t2688
	t2692 = t2690*t2691
	t2693 = t2689 + t2692
	t2695 = t2690*t2687
	t2696 = -1.*t2688*t2691
	t2697 = t2695 + t2696
	t2698 = math.sin(var1[4])
	t2682 = var3[0] + t2681		# p(1) - p(2)
	t2685 = -1.*t2684
	t2686 = 1. + t2685
	t2694 = -0.4*t2686*t2693
	t2699 = 0.4*t2697*t2698
	t2700 = t2684*t2693
	t2701 = t2697*t2698
	t2702 = t2700 + t2701
	t2703 = -0.8*t2702
	t2704 = t2681 + t2694 + t2699 + t2703
	t2709 = 1/t2682
	t2710 = -1.*t2709*t2704		# -tau
	return -t2710	

def tau_Left(var1,var3):
	p_output1 = np.zeros(4)
	t11611 = -1.*var3[1]
	t11624 = math.cos(var1[2])
	t11618 = math.cos(var1[5])
	t11619 = math.sin(var1[2])
	t11625 = math.sin(var1[5])
	t11615 = math.cos(var1[6])
	t11623 = t11618*t11619
	t11627 = t11624*t11625
	t11630 = t11623 + t11627
	t11632 = t11624*t11618
	t11633 = -1.*t11619*t11625
	t11637 = t11632 + t11633
	t11638 = math.sin(var1[6])
	t11612 = var3[0] + t11611
	t11616 = -1.*t11615
	t11617 = 1. + t11616
	t11631 = -0.4*t11617*t11630
	t11639 = 0.4*t11637*t11638
	t11640 = t11615*t11630
	t11643 = t11637*t11638
	t11644 = t11640 + t11643
	t11645 = -0.8*t11644
	t11646 = t11611 + t11631 + t11639 + t11645
	t11653 = 1/t11612
	t11654 = -1.*t11653*t11646 #  -tau
	return -t11654	

def yd_time_RightStance(var1, var2, var3):
	p_output1 = np.zeros(4)
	t2681 = -1.*var3[1]		# -p(2)
	t2690 = math.cos(var1[2])
	t2687 = math.cos(var1[3])
	t2688 = math.sin(var1[2])
	t2691 = math.sin(var1[3])
	t2684 = math.cos(var1[4])
	t2689 = t2687*t2688
	t2692 = t2690*t2691
	t2693 = t2689 + t2692
	t2695 = t2690*t2687
	t2696 = -1.*t2688*t2691
	t2697 = t2695 + t2696
	t2698 = math.sin(var1[4])
	t2682 = var3[0] + t2681		# p(1) - p(2)
	t2685 = -1.*t2684
	t2686 = 1. + t2685
	t2694 = -0.4*t2686*t2693
	t2699 = 0.4*t2697*t2698
	t2700 = t2684*t2693
	t2701 = t2697*t2698
	t2702 = t2700 + t2701
	t2703 = -0.8*t2702
	t2704 = t2681 + t2694 + t2699 + t2703
	t2709 = 1/t2682
	t2710 = -1.*t2709*t2704		# -tau
	t2711 = 1. + t2710			# 1 - tau
	t2683 = math.pow(t2682,-5)
	t2705 = math.pow(t2704,5)
	t2707 = math.pow(t2682,-4)
	t2708 = math.pow(t2704,4)
	t2713 = math.pow(t2682,-3)
	t2714 = math.pow(t2704,3)
	t2715 = math.pow(t2711,2)
	t2717 = math.pow(t2682,-2)
	t2718 = math.pow(t2704,2)
	t2719 = math.pow(t2711,3)
	t2721 = math.pow(t2711,4)
	t2723 = math.pow(t2711,5)
	p_output1[0]=t2723*var2[0] + 5.*t2704*t2709*t2721*var2[4] + 10.*t2717*t2718*t2719*var2[8] + 10.*t2713*t2714*t2715*var2[12] + 5.*t2707*t2708*t2711*var2[16] + t2683*t2705*var2[20]
	p_output1[1]=t2723*var2[1] + 5.*t2704*t2709*t2721*var2[5] + 10.*t2717*t2718*t2719*var2[9] + 10.*t2713*t2714*t2715*var2[13] + 5.*t2707*t2708*t2711*var2[17] + t2683*t2705*var2[21]
	p_output1[2]=t2723*var2[2] + 5.*t2704*t2709*t2721*var2[6] + 10.*t2717*t2718*t2719*var2[10] + 10.*t2713*t2714*t2715*var2[14] + 5.*t2707*t2708*t2711*var2[18] + t2683*t2705*var2[22]
	p_output1[3]=t2723*var2[3] + 5.*t2704*t2709*t2721*var2[7] + 10.*t2717*t2718*t2719*var2[11] + 10.*t2713*t2714*t2715*var2[15] + 5.*t2707*t2708*t2711*var2[19] + t2683*t2705*var2[23]
	return p_output1, -t2710

def yd_time_LeftStance(var1, var2, var3):
	p_output1 = np.zeros(4)
	t11611 = -1.*var3[1]
	t11624 = math.cos(var1[2])
	t11618 = math.cos(var1[5])
	t11619 = math.sin(var1[2])
	t11625 = math.sin(var1[5])
	t11615 = math.cos(var1[6])
	t11623 = t11618*t11619
	t11627 = t11624*t11625
	t11630 = t11623 + t11627
	t11632 = t11624*t11618
	t11633 = -1.*t11619*t11625
	t11637 = t11632 + t11633
	t11638 = math.sin(var1[6])
	t11612 = var3[0] + t11611
	t11616 = -1.*t11615
	t11617 = 1. + t11616
	t11631 = -0.4*t11617*t11630
	t11639 = 0.4*t11637*t11638
	t11640 = t11615*t11630
	t11643 = t11637*t11638
	t11644 = t11640 + t11643
	t11645 = -0.8*t11644
	t11646 = t11611 + t11631 + t11639 + t11645
	t11653 = 1/t11612
	t11654 = -1.*t11653*t11646	# -tau
	t11656 = 1. + t11654
	t11613 = math.pow(t11612,-5)
	t11649 = math.pow(t11646,5)
	t11651 = math.pow(t11612,-4)
	t11652 = math.pow(t11646,4)
	t11661 = math.pow(t11612,-3)
	t11662 = math.pow(t11646,3)
	t11664 = math.pow(t11656,2)
	t11666 = math.pow(t11612,-2)
	t11667 = math.pow(t11646,2)
	t11699 = math.pow(t11656,3)
	t11705 = math.pow(t11656,4)
	t11708 = math.pow(t11656,5)
	p_output1[0]=t11708*var2[0] + 5.*t11646*t11653*t11705*var2[4] + 10.*t11666*t11667*t11699*var2[8] + 10.*t11661*t11662*t11664*var2[12] + 5.*t11651*t11652*t11656*var2[16] + t11613*t11649*var2[20]
	p_output1[1]=t11708*var2[1] + 5.*t11646*t11653*t11705*var2[5] + 10.*t11666*t11667*t11699*var2[9] + 10.*t11661*t11662*t11664*var2[13] + 5.*t11651*t11652*t11656*var2[17] + t11613*t11649*var2[21]
	p_output1[2]=t11708*var2[2] + 5.*t11646*t11653*t11705*var2[6] + 10.*t11666*t11667*t11699*var2[10] + 10.*t11661*t11662*t11664*var2[14] + 5.*t11651*t11652*t11656*var2[18] + t11613*t11649*var2[22]
	p_output1[3]=t11708*var2[3] + 5.*t11646*t11653*t11705*var2[7] + 10.*t11666*t11667*t11699*var2[11] + 10.*t11661*t11662*t11664*var2[15] + 5.*t11651*t11652*t11656*var2[19] + t11613*t11649*var2[23]
	return p_output1

def d1yd_time_RightStance(var1, var2, var3, var4):
	p_output1 = np.zeros(4)
	t2726 = math.cos(var1[3])
	t2728 = math.sin(var1[2])
	t2725 = math.cos(var1[2])
	t2729 = math.sin(var1[3])
	t2720 = math.cos(var1[4])
	t2727 = t2725*t2726
	t2730 = -1.*t2728*t2729
	t2731 = t2727 + t2730
	t2733 = -1.*t2726*t2728
	t2734 = -1.*t2725*t2729
	t2735 = t2733 + t2734
	t2736 = math.sin(var1[4])
	t2706 = -1.*var4[1]
	t2722 = -1.*t2720
	t2724 = 1. + t2722
	t2743 = t2726*t2728
	t2744 = t2725*t2729
	t2745 = t2743 + t2744
	t2712 = var4[0] + t2706
	t2716 = math.pow(t2712,-5)
	t2732 = -0.4*t2724*t2731
	t2737 = 0.4*t2735*t2736
	t2738 = t2720*t2731
	t2739 = t2735*t2736
	t2740 = t2738 + t2739
	t2741 = -0.8*t2740
	t2742 = t2732 + t2737 + t2741
	t2746 = -0.4*t2724*t2745
	t2747 = 0.4*t2731*t2736
	t2748 = t2720*t2745
	t2749 = t2731*t2736
	t2750 = t2748 + t2749
	t2751 = -0.8*t2750
	t2752 = t2706 + t2746 + t2747 + t2751
	t2753 = math.pow(t2752,4)
	t2756 = math.pow(t2712,-4)
	t2757 = math.pow(t2752,3)
	t2758 = 1/t2712
	t2759 = -1.*t2758*t2752
	t2760 = 1. + t2759
	t2763 = math.pow(t2712,-3)
	t2764 = math.pow(t2752,2)
	t2765 = math.pow(t2760,2)
	t2768 = math.pow(t2712,-2)
	t2769 = math.pow(t2760,3)
	t2772 = math.pow(t2760,4)
	t2754 = -5.*var3[16]*t2716*t2742*t2753
	t2755 = 5.*var3[20]*t2716*t2742*t2753
	t2761 = -20.*var3[12]*t2756*t2742*t2757*t2760
	t2762 = 20.*var3[16]*t2756*t2742*t2757*t2760
	t2766 = -30.*var3[8]*t2763*t2742*t2764*t2765
	t2767 = 30.*var3[12]*t2763*t2742*t2764*t2765
	t2770 = -20.*var3[4]*t2768*t2742*t2752*t2769
	t2771 = 20.*var3[8]*t2768*t2742*t2752*t2769
	t2773 = -5.*var3[0]*t2758*t2742*t2772
	t2774 = 5.*var3[4]*t2758*t2742*t2772
	t2775 = t2754 + t2755 + t2761 + t2762 + t2766 + t2767 + t2770 + t2771 + t2773 + t2774
	t2778 = 0.4*t2720*t2731
	t2779 = -0.4*t2745*t2736
	t2780 = -1.*t2745*t2736
	t2781 = t2738 + t2780
	t2782 = -0.8*t2781
	t2783 = t2778 + t2779 + t2782
	t2797 = -5.*var3[17]*t2716*t2742*t2753
	t2798 = 5.*var3[21]*t2716*t2742*t2753
	t2799 = -20.*var3[13]*t2756*t2742*t2757*t2760
	t2800 = 20.*var3[17]*t2756*t2742*t2757*t2760
	t2801 = -30.*var3[9]*t2763*t2742*t2764*t2765
	t2802 = 30.*var3[13]*t2763*t2742*t2764*t2765
	t2803 = -20.*var3[5]*t2768*t2742*t2752*t2769
	t2804 = 20.*var3[9]*t2768*t2742*t2752*t2769
	t2805 = -5.*var3[1]*t2758*t2742*t2772
	t2806 = 5.*var3[5]*t2758*t2742*t2772
	t2807 = t2797 + t2798 + t2799 + t2800 + t2801 + t2802 + t2803 + t2804 + t2805 + t2806
	t2823 = -5.*var3[18]*t2716*t2742*t2753
	t2824 = 5.*var3[22]*t2716*t2742*t2753
	t2825 = -20.*var3[14]*t2756*t2742*t2757*t2760
	t2826 = 20.*var3[18]*t2756*t2742*t2757*t2760
	t2827 = -30.*var3[10]*t2763*t2742*t2764*t2765
	t2828 = 30.*var3[14]*t2763*t2742*t2764*t2765
	t2829 = -20.*var3[6]*t2768*t2742*t2752*t2769
	t2830 = 20.*var3[10]*t2768*t2742*t2752*t2769
	t2831 = -5.*var3[2]*t2758*t2742*t2772
	t2832 = 5.*var3[6]*t2758*t2742*t2772
	t2833 = t2823 + t2824 + t2825 + t2826 + t2827 + t2828 + t2829 + t2830 + t2831 + t2832
	t2849 = -5.*var3[19]*t2716*t2742*t2753
	t2850 = 5.*var3[23]*t2716*t2742*t2753
	t2851 = -20.*var3[15]*t2756*t2742*t2757*t2760
	t2852 = 20.*var3[19]*t2756*t2742*t2757*t2760
	t2853 = -30.*var3[11]*t2763*t2742*t2764*t2765
	t2854 = 30.*var3[15]*t2763*t2742*t2764*t2765
	t2855 = -20.*var3[7]*t2768*t2742*t2752*t2769
	t2856 = 20.*var3[11]*t2768*t2742*t2752*t2769
	t2857 = -5.*var3[3]*t2758*t2742*t2772
	t2858 = 5.*var3[7]*t2758*t2742*t2772
	t2859 = t2849 + t2850 + t2851 + t2852 + t2853 + t2854 + t2855 + t2856 + t2857 + t2858
	p_output1[0]=t2775*var2[2] + t2775*var2[3] + var2[4]*(-5.*t2758*t2772*t2783*var3[0] - 20.*t2752*t2768*t2769*t2783*var3[4] + 5.*t2758*t2772*t2783*var3[4] - 30.*t2763*t2764*t2765*t2783*var3[8] + 20.*t2752*t2768*t2769*t2783*var3[8] - 20.*t2756*t2757*t2760*t2783*var3[12] + 30.*t2763*t2764*t2765*t2783*var3[12] - 5.*t2716*t2753*t2783*var3[16] + 20.*t2756*t2757*t2760*t2783*var3[16] + 5.*t2716*t2753*t2783*var3[20])
	p_output1[1]=t2807*var2[2] + t2807*var2[3] + var2[4]*(-5.*t2758*t2772*t2783*var3[1] - 20.*t2752*t2768*t2769*t2783*var3[5] + 5.*t2758*t2772*t2783*var3[5] - 30.*t2763*t2764*t2765*t2783*var3[9] + 20.*t2752*t2768*t2769*t2783*var3[9] - 20.*t2756*t2757*t2760*t2783*var3[13] + 30.*t2763*t2764*t2765*t2783*var3[13] - 5.*t2716*t2753*t2783*var3[17] + 20.*t2756*t2757*t2760*t2783*var3[17] + 5.*t2716*t2753*t2783*var3[21])
	p_output1[2]=t2833*var2[2] + t2833*var2[3] + var2[4]*(-5.*t2758*t2772*t2783*var3[2] - 20.*t2752*t2768*t2769*t2783*var3[6] + 5.*t2758*t2772*t2783*var3[6] - 30.*t2763*t2764*t2765*t2783*var3[10] + 20.*t2752*t2768*t2769*t2783*var3[10] - 20.*t2756*t2757*t2760*t2783*var3[14] + 30.*t2763*t2764*t2765*t2783*var3[14] - 5.*t2716*t2753*t2783*var3[18] + 20.*t2756*t2757*t2760*t2783*var3[18] + 5.*t2716*t2753*t2783*var3[22])
	p_output1[3]=t2859*var2[2] + t2859*var2[3] + var2[4]*(-5.*t2758*t2772*t2783*var3[3] - 20.*t2752*t2768*t2769*t2783*var3[7] + 5.*t2758*t2772*t2783*var3[7] - 30.*t2763*t2764*t2765*t2783*var3[11] + 20.*t2752*t2768*t2769*t2783*var3[11] - 20.*t2756*t2757*t2760*t2783*var3[15] + 30.*t2763*t2764*t2765*t2783*var3[15] - 5.*t2716*t2753*t2783*var3[19] + 20.*t2756*t2757*t2760*t2783*var3[19] + 5.*t2716*t2753*t2783*var3[23])
	return p_output1

def d1yd_time_LeftStance(var1, var2, var3, var4):
	p_output1 = np.zeros(4)
	t3456 = math.cos(var1[5])
	t3458 = math.sin(var1[2])
	t3455 = math.cos(var1[2])
	t3459 = math.sin(var1[5])
	t3450 = math.cos(var1[6])
	t3457 = t3455*t3456
	t3460 = -1.*t3458*t3459
	t3461 = t3457 + t3460
	t3463 = -1.*t3456*t3458
	t3464 = -1.*t3455*t3459
	t3465 = t3463 + t3464
	t3466 = math.sin(var1[6])
	t3436 = -1.*var4[1]
	t3452 = -1.*t3450
	t3454 = 1. + t3452
	t3473 = t3456*t3458
	t3474 = t3455*t3459
	t3475 = t3473 + t3474
	t3442 = var4[0] + t3436
	t3446 = math.pow(t3442,-5)
	t3462 = -0.4*t3454*t3461
	t3467 = 0.4*t3465*t3466
	t3468 = t3450*t3461
	t3469 = t3465*t3466
	t3470 = t3468 + t3469
	t3471 = -0.8*t3470
	t3472 = t3462 + t3467 + t3471
	t3476 = -0.4*t3454*t3475
	t3477 = 0.4*t3461*t3466
	t3478 = t3450*t3475
	t3479 = t3461*t3466
	t3480 = t3478 + t3479
	t3481 = -0.8*t3480
	t3482 = t3436 + t3476 + t3477 + t3481
	t3483 = math.pow(t3482,4)
	t3486 = math.pow(t3442,-4)
	t3487 = math.pow(t3482,3)
	t3488 = 1/t3442
	t3489 = -1.*t3488*t3482
	t3490 = 1. + t3489
	t3493 = math.pow(t3442,-3)
	t3494 = math.pow(t3482,2)
	t3495 = math.pow(t3490,2)
	t3498 = math.pow(t3442,-2)
	t3499 = math.pow(t3490,3)
	t3502 = math.pow(t3490,4)
	t3484 = -5.*var3[16]*t3446*t3472*t3483
	t3485 = 5.*var3[20]*t3446*t3472*t3483
	t3491 = -20.*var3[12]*t3486*t3472*t3487*t3490
	t3492 = 20.*var3[16]*t3486*t3472*t3487*t3490
	t3496 = -30.*var3[8]*t3493*t3472*t3494*t3495
	t3497 = 30.*var3[12]*t3493*t3472*t3494*t3495
	t3500 = -20.*var3[4]*t3498*t3472*t3482*t3499
	t3501 = 20.*var3[8]*t3498*t3472*t3482*t3499
	t3503 = -5.*var3[0]*t3488*t3472*t3502
	t3504 = 5.*var3[4]*t3488*t3472*t3502
	t3505 = t3484 + t3485 + t3491 + t3492 + t3496 + t3497 + t3500 + t3501 + t3503 + t3504
	t3508 = 0.4*t3450*t3461
	t3509 = -0.4*t3475*t3466
	t3510 = -1.*t3475*t3466
	t3511 = t3468 + t3510
	t3512 = -0.8*t3511
	t3513 = t3508 + t3509 + t3512
	t3527 = -5.*var3[17]*t3446*t3472*t3483
	t3528 = 5.*var3[21]*t3446*t3472*t3483
	t3529 = -20.*var3[13]*t3486*t3472*t3487*t3490
	t3530 = 20.*var3[17]*t3486*t3472*t3487*t3490
	t3531 = -30.*var3[9]*t3493*t3472*t3494*t3495
	t3532 = 30.*var3[13]*t3493*t3472*t3494*t3495
	t3533 = -20.*var3[5]*t3498*t3472*t3482*t3499
	t3534 = 20.*var3[9]*t3498*t3472*t3482*t3499
	t3535 = -5.*var3[1]*t3488*t3472*t3502
	t3536 = 5.*var3[5]*t3488*t3472*t3502
	t3537 = t3527 + t3528 + t3529 + t3530 + t3531 + t3532 + t3533 + t3534 + t3535 + t3536
	t3553 = -5.*var3[18]*t3446*t3472*t3483
	t3554 = 5.*var3[22]*t3446*t3472*t3483
	t3555 = -20.*var3[14]*t3486*t3472*t3487*t3490
	t3556 = 20.*var3[18]*t3486*t3472*t3487*t3490
	t3557 = -30.*var3[10]*t3493*t3472*t3494*t3495
	t3558 = 30.*var3[14]*t3493*t3472*t3494*t3495
	t3559 = -20.*var3[6]*t3498*t3472*t3482*t3499
	t3560 = 20.*var3[10]*t3498*t3472*t3482*t3499
	t3561 = -5.*var3[2]*t3488*t3472*t3502
	t3562 = 5.*var3[6]*t3488*t3472*t3502
	t3563 = t3553 + t3554 + t3555 + t3556 + t3557 + t3558 + t3559 + t3560 + t3561 + t3562
	t3579 = -5.*var3[19]*t3446*t3472*t3483
	t3580 = 5.*var3[23]*t3446*t3472*t3483
	t3581 = -20.*var3[15]*t3486*t3472*t3487*t3490
	t3582 = 20.*var3[19]*t3486*t3472*t3487*t3490
	t3583 = -30.*var3[11]*t3493*t3472*t3494*t3495
	t3584 = 30.*var3[15]*t3493*t3472*t3494*t3495
	t3585 = -20.*var3[7]*t3498*t3472*t3482*t3499
	t3586 = 20.*var3[11]*t3498*t3472*t3482*t3499
	t3587 = -5.*var3[3]*t3488*t3472*t3502
	t3588 = 5.*var3[7]*t3488*t3472*t3502
	t3589 = t3579 + t3580 + t3581 + t3582 + t3583 + t3584 + t3585 + t3586 + t3587 + t3588
	p_output1[0]=t3505*var2[2] + t3505*var2[5] + var2[6]*(-5.*t3488*t3502*t3513*var3[0] - 20.*t3482*t3498*t3499*t3513*var3[4] + 5.*t3488*t3502*t3513*var3[4] - 30.*t3493*t3494*t3495*t3513*var3[8] + 20.*t3482*t3498*t3499*t3513*var3[8] - 20.*t3486*t3487*t3490*t3513*var3[12] + 30.*t3493*t3494*t3495*t3513*var3[12] - 5.*t3446*t3483*t3513*var3[16] + 20.*t3486*t3487*t3490*t3513*var3[16] + 5.*t3446*t3483*t3513*var3[20])
	p_output1[1]=t3537*var2[2] + t3537*var2[5] + var2[6]*(-5.*t3488*t3502*t3513*var3[1] - 20.*t3482*t3498*t3499*t3513*var3[5] + 5.*t3488*t3502*t3513*var3[5] - 30.*t3493*t3494*t3495*t3513*var3[9] + 20.*t3482*t3498*t3499*t3513*var3[9] - 20.*t3486*t3487*t3490*t3513*var3[13] + 30.*t3493*t3494*t3495*t3513*var3[13] - 5.*t3446*t3483*t3513*var3[17] + 20.*t3486*t3487*t3490*t3513*var3[17] + 5.*t3446*t3483*t3513*var3[21])
	p_output1[2]=t3563*var2[2] + t3563*var2[5] + var2[6]*(-5.*t3488*t3502*t3513*var3[2] - 20.*t3482*t3498*t3499*t3513*var3[6] + 5.*t3488*t3502*t3513*var3[6] - 30.*t3493*t3494*t3495*t3513*var3[10] + 20.*t3482*t3498*t3499*t3513*var3[10] - 20.*t3486*t3487*t3490*t3513*var3[14] + 30.*t3493*t3494*t3495*t3513*var3[14] - 5.*t3446*t3483*t3513*var3[18] + 20.*t3486*t3487*t3490*t3513*var3[18] + 5.*t3446*t3483*t3513*var3[22])
	p_output1[3]=t3589*var2[2] + t3589*var2[5] + var2[6]*(-5.*t3488*t3502*t3513*var3[3] - 20.*t3482*t3498*t3499*t3513*var3[7] + 5.*t3488*t3502*t3513*var3[7] - 30.*t3493*t3494*t3495*t3513*var3[11] + 20.*t3482*t3498*t3499*t3513*var3[11] - 20.*t3486*t3487*t3490*t3513*var3[15] + 30.*t3493*t3494*t3495*t3513*var3[15] - 5.*t3446*t3483*t3513*var3[19] + 20.*t3486*t3487*t3490*t3513*var3[19] + 5.*t3446*t3483*t3513*var3[23])
	return p_output1

def left_foot_height(var1):
	p_output = 0.4*math.cos(var1[2]+var1[5]) + 0.4*math.cos(var1[2]+var1[5]+var1[6]) + var1[1]
	return p_output

def right_foot_height(var1):
	p_output = 0.4*math.cos(var1[2]+var1[3]) + 0.4*math.cos(var1[2]+var1[3]+var1[4]) + var1[1]
	return p_output