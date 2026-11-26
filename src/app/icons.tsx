import type { SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement> & { size?: number };

function Icon({ size = 15, className, children, ...rest }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 20 20"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className={className ? `yc-icon ${className}` : "yc-icon"}
      {...rest}
    >
      {children}
    </svg>
  );
}

export function IconPlus(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M10 4.25v11.5M4.25 10h11.5" />
    </Icon>
  );
}

export function IconClose(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M5.25 5.25l9.5 9.5M14.75 5.25l-9.5 9.5" />
    </Icon>
  );
}

export function IconChevronDown(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M5.5 7.75L10 12.25l4.5-4.5" />
    </Icon>
  );
}

export function IconChevronLeft(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M12.25 5.5L7.75 10l4.5 4.5" />
    </Icon>
  );
}

export function IconChevronRight(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M7.75 5.5L12.25 10l-4.5 4.5" />
    </Icon>
  );
}

export function IconArrowLeft(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M15.5 10H4.75M9 5.75L4.75 10 9 14.25" />
    </Icon>
  );
}

export function IconWaveform(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M3.25 8.5v3M6.5 5.5v9M10 3.25v13.5M13.5 6v8M16.75 8.5v3" />
    </Icon>
  );
}

export function IconBolt(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M11.2 2.75L4.75 11h4.8l-.75 6.25L15.25 9h-4.8l.75-6.25z" />
    </Icon>
  );
}

export function IconSparkle(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M10 2.75l1.65 4.6 4.6 1.65-4.6 1.65L10 15.25l-1.65-4.6-4.6-1.65 4.6-1.65L10 2.75z" />
    </Icon>
  );
}

export function IconHistory(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M3.75 10a6.25 6.25 0 1 0 1.83-4.42L3.5 7.5" />
      <path d="M3.5 3.75v3.75h3.75" />
      <path d="M10 6.75V10l2.5 1.5" />
    </Icon>
  );
}

export function IconSliders(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M3.25 5.5h13.5M3.25 10h13.5M3.25 14.5h13.5" />
      <circle cx="7" cy="5.5" r="1.65" fill="var(--panel, #131620)" />
      <circle cx="13" cy="10" r="1.65" fill="var(--panel, #131620)" />
      <circle cx="8.5" cy="14.5" r="1.65" fill="var(--panel, #131620)" />
    </Icon>
  );
}

export function IconMore(props: IconProps) {
  return (
    <Icon {...props}>
      <circle cx="5" cy="10" r="1.15" fill="currentColor" stroke="none" />
      <circle cx="10" cy="10" r="1.15" fill="currentColor" stroke="none" />
      <circle cx="15" cy="10" r="1.15" fill="currentColor" stroke="none" />
    </Icon>
  );
}

export function IconMic(props: IconProps) {
  return (
    <Icon {...props}>
      <rect x="7.25" y="2.75" width="5.5" height="9" rx="2.75" />
      <path d="M4.75 9.5a5.25 5.25 0 0 0 10.5 0M10 14.75v2.5M7.25 17.25h5.5" />
    </Icon>
  );
}

export function IconSoundwave(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M3.25 7.75h3l3.75-3.25v11L6.25 12.25h-3a.75.75 0 0 1-.75-.75v-3a.75.75 0 0 1 .75-.75z" />
      <path d="M13.25 7.25a4 4 0 0 1 0 5.5M15.65 5a7 7 0 0 1 0 10" />
    </Icon>
  );
}

export function IconDocument(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M5.25 2.75h6.5l3 3v11.5a.75.75 0 0 1-.75.75H5.25a.75.75 0 0 1-.75-.75V3.5a.75.75 0 0 1 .75-.75z" />
      <path d="M11.5 2.75v3.25h3.25M7.25 9.5h5.5M7.25 12.75h4" />
    </Icon>
  );
}

export function IconLab(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M7.5 3h5M8.5 3v4.6L4.6 15a1.5 1.5 0 0 0 1.32 2.25h8.16A1.5 1.5 0 0 0 15.4 15l-3.9-7.4V3" />
      <path d="M6.4 12.5h7.2" />
    </Icon>
  );
}

export function IconFolderPlus(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M2.75 5.25a1.5 1.5 0 0 1 1.5-1.5h3.4l1.6 1.75h6.5a1.5 1.5 0 0 1 1.5 1.5v7.75a1.5 1.5 0 0 1-1.5 1.5H4.25a1.5 1.5 0 0 1-1.5-1.5V5.25z" />
      <path d="M10 8.5v4.5M7.75 10.75h4.5" />
    </Icon>
  );
}

export function IconPlay({ size = 14, className, ...rest }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 20 20"
      fill="currentColor"
      aria-hidden="true"
      className={className ? `yc-icon ${className}` : "yc-icon"}
      {...rest}
    >
      <path d="M6.4 4.35a.9.9 0 0 1 1.37-.77l8.55 5.65a.9.9 0 0 1 0 1.54l-8.55 5.65a.9.9 0 0 1-1.37-.77V4.35z" />
    </svg>
  );
}

export function IconPause({ size = 14, className, ...rest }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 20 20"
      fill="currentColor"
      aria-hidden="true"
      className={className ? `yc-icon ${className}` : "yc-icon"}
      {...rest}
    >
      <rect x="5.25" y="4" width="3.35" height="12" rx="1.1" />
      <rect x="11.4" y="4" width="3.35" height="12" rx="1.1" />
    </svg>
  );
}

export function IconStop({ size = 14, className, ...rest }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 20 20"
      fill="currentColor"
      aria-hidden="true"
      className={className ? `yc-icon ${className}` : "yc-icon"}
      {...rest}
    >
      <rect x="5" y="5" width="10" height="10" rx="2" />
    </svg>
  );
}

export function IconTrash(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M4.25 5.5h11.5M7.75 5.5V4a1 1 0 0 1 1-1h2.5a1 1 0 0 1 1 1v1.5" />
      <path d="M5.75 5.5l.75 10.25a1.5 1.5 0 0 0 1.5 1.4h4a1.5 1.5 0 0 0 1.5-1.4l.75-10.25" />
      <path d="M8.5 8.75v5M11.5 8.75v5" />
    </Icon>
  );
}

export function IconCheck(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M4.5 10.5l3.6 3.6 7.4-7.85" />
    </Icon>
  );
}

export function IconCheckCircle(props: IconProps) {
  return (
    <Icon {...props}>
      <circle cx="10" cy="10" r="7" />
      <path d="M6.8 10.2l2.15 2.15 4.3-4.6" />
    </Icon>
  );
}

export function IconInfoCircle(props: IconProps) {
  return (
    <Icon {...props}>
      <circle cx="10" cy="10" r="7" />
      <path d="M10 9.25v4M10 6.75h.01" />
    </Icon>
  );
}

export function IconDownload(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M10 3.25v9.25M6.5 9.25L10 12.75l3.5-3.5" />
      <path d="M3.75 14.25v1.5a1 1 0 0 0 1 1h10.5a1 1 0 0 0 1-1v-1.5" />
    </Icon>
  );
}

export function IconUpload(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M10 12.75V3.5M6.5 6.75L10 3.25l3.5 3.5" />
      <path d="M3.75 14.25v1.5a1 1 0 0 0 1 1h10.5a1 1 0 0 0 1-1v-1.5" />
    </Icon>
  );
}

export function IconSwap(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M4 7h11.5M12.25 4L15.5 7l-3.25 3" />
      <path d="M16 13H4.5M7.75 10L4.5 13l3.25 3" />
    </Icon>
  );
}

export function IconFilter(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M3.5 5.25h13M5.75 10h8.5M8.25 14.75h3.5" />
    </Icon>
  );
}

export function IconDirector(props: IconProps) {
  return (
    <Icon {...props}>
      <rect x="3" y="4.25" width="14" height="11.5" rx="2" />
      <path d="M3 8.25h14M7 4.25l-1.5 4M12 4.25l-1.5 4" />
    </Icon>
  );
}

export function IconClock(props: IconProps) {
  return (
    <Icon {...props}>
      <circle cx="10" cy="10" r="7" />
      <path d="M10 6.25V10l2.75 1.75" />
    </Icon>
  );
}

export function IconShield(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M10 2.75l6 2.5v4.5c0 3.9-2.55 6.45-6 7.75-3.45-1.3-6-3.85-6-7.75v-4.5l6-2.5z" />
      <path d="M7.5 10.2l1.8 1.8 3.5-3.8" />
    </Icon>
  );
}

export function IconShuffle(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M3.25 6h2.6a3 3 0 0 1 2.4 1.2l3.5 4.6a3 3 0 0 0 2.4 1.2h2.6" />
      <path d="M14.5 4l2.25 2-2.25 2M14.5 11l2.25 2-2.25 2" />
      <path d="M3.25 13h2.6a3 3 0 0 0 2.15-.9M12 6.9a3 3 0 0 1 2.15-.9h2.6" />
    </Icon>
  );
}

export function IconRepeat(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M14.25 4.25L16.75 6.5l-2.5 2.25" />
      <path d="M4 10.5V8.75A2.25 2.25 0 0 1 6.25 6.5h10.5" />
      <path d="M5.75 15.75L3.25 13.5l2.5-2.25" />
      <path d="M16 9.5v1.75a2.25 2.25 0 0 1-2.25 2.25H3.25" />
    </Icon>
  );
}

export function IconLayers(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M10 3.25L3.25 7 10 10.75 16.75 7 10 3.25z" />
      <path d="M3.25 10.5L10 14.25l6.75-3.75" />
      <path d="M3.25 13.75L10 17.5l6.75-3.75" />
    </Icon>
  );
}
